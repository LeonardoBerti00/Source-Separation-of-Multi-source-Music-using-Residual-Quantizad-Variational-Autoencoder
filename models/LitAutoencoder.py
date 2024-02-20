import os
import time
import torch
import wandb
import lightning as L
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from config import Configuration
from constants import LearningHyperParameter
from torch.nn import functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import constants as cst
from models.VQVAE.VQVAE import VQVAE
from models.VAE.VAE import VAE
from models.RQVAE.RQVAE import RQVAE
from utils.utils import save_audio


class LitAutoencoder(L.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()

        self.IS_WANDB = config.IS_WANDB
        self.IS_DEBUG = config.IS_DEBUG
        self.chosen_model = config.CHOSEN_MODEL
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.codebook_length = config.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH]
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.is_training = config.IS_TRAINING
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.val_snrs, self.test_snrs = [], []
        self.min_loss = 10000000
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.last_path_ckpt = None
        if self.IS_WANDB:
            self.filename_ckpt = config.FILENAME_CKPT
            self.save_hyperparameters()

        self.AE = self.pick_autoencoder(config.CHOSEN_AE.name, config)
        

    def forward(self, x, is_train, batch_idx):
        return self.AE.forward(x, is_train, batch_idx)


    def loss(self, input, recon, quantizations, residuals):
        return self.AE.loss(input, recon, quantizations, residuals)


    def training_step(self, x, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(x, dim=-2)
        mixture.requires_grad = True
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        recon, quantizations, residuals = self.forward(mixture, is_train=True, batch_idx=batch_idx)
        batch_losses = self.loss(x, recon, quantizations, residuals)
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        #self.ema.update()
        if self.IS_DEBUG and batch_idx % 10 == 0:
            print(f"\nrecon mean: {recon.mean()}")
            print(f"input mean: {x.mean()}")
        
        return batch_loss_mean


    def on_validation_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch+1)
            wandb.log({'recon_time_term': sum(self.AE.recon_time_terms) / len(self.AE.recon_time_terms)}, step=self.current_epoch+1)
            wandb.log({'multi_spectral_recon_loss': sum(self.AE.multi_spectral_recon_losses) / len(self.AE.multi_spectral_recon_losses)}, step=self.current_epoch+1)
            wandb.log({'commitment_loss': sum(self.AE.commitment_losses) / len(self.AE.commitment_losses)}, step=self.current_epoch+1)
        self.AE.recon_time_terms = []
        self.AE.multi_spectral_recon_losses = []
        self.AE.commitment_losses = []
        self.train_losses = []
        self.val_ema_losses = []
        self.val_losses = []
        print(f'\ntrain loss on epoch {self.current_epoch} is {round(loss, 4)}')


    def validation_step(self, x, batch_idx):
        # from source to mixture
        mixture = torch.sum(x, dim=-2)
        recon, quantizations, residuals = self.forward(mixture, is_train=False, batch_idx=batch_idx)
        batch_losses = self.loss(x, recon, quantizations, residuals)
        batch_loss_mean = torch.mean(batch_losses)
        self.val_losses.append(batch_loss_mean.item())
        self.val_snrs.append(self.si_snr(x, recon).item())
        '''
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, z_q, z_e = self.forward(mixture, is_train=False)
            batch_ema_losses = self.loss(x, recon, z_q, z_e)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        '''
        if batch_idx % 10 == 0 and self.IS_DEBUG:
            save_audio(recon[0][0], 'recon_bass'+str(batch_idx))
            save_audio(recon[0][1], 'recon_drums'+str(batch_idx))
            save_audio(recon[0][2], 'recon_guitar'+str(batch_idx))
            save_audio(recon[0][3], 'recon_piano'+str(batch_idx))
            save_audio(x[0][0], 'source_bass'+str(batch_idx))
            save_audio(x[0][1], 'source_drums'+str(batch_idx))
            save_audio(x[0][2], 'source_guitar'+str(batch_idx))
            save_audio(x[0][3], 'source_piano'+str(batch_idx))
            print(f"recon mean: {recon.mean()}")
            print(f"input mean: {x.mean()}") 
        return batch_loss_mean


    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_snr = sum(self.val_snrs) / len(self.val_snrs)
        #loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', self.val_loss)
        #self.log('val_ema_loss', loss_ema)
        self.log('val_snr', self.val_snr)
        # model checkpointing
        if self.val_snr < self.min_loss:
            self.min_loss = self.val_snr
            if self.IS_WANDB:
                self._model_checkpointing(self.val_snr)
        print(f"\n val loss on epoch {self.current_epoch} is {round(self.val_loss, 4)}")
        #print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")
        print(f"\n val snr on epoch {self.current_epoch} is {round(self.val_snr, 4)}")


    def test_step(self, x, batch_idx):
        # from source to mixture
        mixture = torch.sum(x, dim=-2)
        recon, quantizations, residuals = self.forward(mixture, is_train=False, batch_idx=batch_idx)
        batch_losses = self.loss(x, recon, quantizations, residuals)
        batch_loss_mean = torch.mean(batch_losses)
        self.test_losses.append(batch_loss_mean.item())
        self.test_snrs.append(self.si_snr(x, recon).item())
        '''
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[
            batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.test_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        
        # Testing: with EMA
        with self.ema.average_parameters():
            recon, z_q, z_e = self.forward(mixture, is_train=False)
            batch_ema_losses = self.loss(x, recon, z_q, z_e)
            ema_loss = torch.mean(batch_ema_losses)
            self.test_ema_losses.append(ema_loss.item())
            
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.test_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
            '''
        #print recon mean
        if self.IS_DEBUG:
            print(f"recon mean: {recon.mean()}")
            print(f"input mean: {x.mean()}")    
        if batch_idx % 10 == 0 and self.IS_DEBUG:
            save_audio(recon[0][0], 'recon_bass'+str(batch_idx))
            save_audio(recon[0][1], 'recon_drums'+str(batch_idx))
            save_audio(recon[0][2], 'recon_guitar'+str(batch_idx))
            save_audio(recon[0][3], 'recon_piano'+str(batch_idx))
            save_audio(x[0][0], 'source_bass'+str(batch_idx))
            save_audio(x[0][1], 'source_drums'+str(batch_idx))
            save_audio(x[0][2], 'source_guitar'+str(batch_idx))
            save_audio(x[0][3], 'source_piano'+str(batch_idx))
        return batch_loss_mean


    def on_test_end(self) -> None:
        test_loss = sum(self.test_losses) / len(self.test_losses)
        #loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        test_snr = sum(self.test_snrs) / len(self.test_snrs)
        if self.IS_WANDB:
            wandb.log({'test_snr': test_snr}, step=self.current_epoch+1)
            wandb.log({'test_loss': test_loss}, step=self.current_epoch+1)
            #wandb.log({'test_ema_loss': loss_ema}, step=self.current_epoch+1)
        print(f"\n test loss on epoch {self.current_epoch} is {round(test_loss, 4)}")
        #print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")
        print(f"\n test snr on epoch {self.current_epoch} is {round(test_snr, 4)}")
        #numpy.save(cst.RECON_DIR + "/test_reconstructions.npy", self.test_reconstructions)
        #numpy.save(cst.RECON_DIR + "/test_ema_reconstructions.npy", self.test_ema_reconstructions)


    def configure_optimizers(self):
        if self.lr == 0.001:
            end_factor = 0.01
        else:
            end_factor = 0.05
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=end_factor, total_iters=self.epochs)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    #def on_before_zero_grad(self, *args, **kwargs):
    #    self.ema.update()    

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        #wandb.define_metric("val_ema_loss", summary="min")
        wandb.define_metric("val_snr", summary="max")

    def inference_time(self, x):
        t0 = time.time()
        _ = self.forward(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed
    
    def _model_checkpointing(self, loss):
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        filename_ckpt = ("val_snr=" + str(round(loss, 4)) +
                             "_epoch=" + str(self.current_epoch) +
                             "_" + self.filename_ckpt
                             )
        path_ckpt = cst.DIR_SAVED_MODEL + "/" + self.chosen_model.name + "/" + filename_ckpt
        #with self.ema.average_parameters():
        self.trainer.save_checkpoint(path_ckpt)
        self.last_path_ckpt = path_ckpt

    def pick_autoencoder(self, autoencoder_name, config):
        if autoencoder_name == "VQVAE":
            return VQVAE(config).to(device=cst.DEVICE)
        elif autoencoder_name == 'VAE':
            return VAE(config).to(device=cst.DEVICE)
        elif autoencoder_name == 'RQVAE':
            return RQVAE(config).to(device=cst.DEVICE)
        else:
            raise ValueError("Autoencoder not found")






