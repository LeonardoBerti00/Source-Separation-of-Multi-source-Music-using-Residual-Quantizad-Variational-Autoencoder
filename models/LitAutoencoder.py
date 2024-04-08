import os
import time
from einops import rearrange, repeat
import numpy as np
import torch
import wandb
import lightning as L
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from config import Configuration
from constants import LearningHyperParameter
from torch.nn import functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio

import constants as cst
from models.VQVAE.VQVAE import VQVAE
from models.VAE.VAE import VAE
from models.RQVAE.RQVAE import RQVAE
from utils.utils import save_audio, is_silent
from vector_quantize_pytorch import ResidualVQ


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
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_sdr, self.test_ema_sdr = [], []
        self.val_ema_sdr2, self.test_ema_sdr2 = [], []
        self.val_snrs, self.test_snrs = [], []
        self.val_sdrs, self.test_sdrs = [], []
        self.test_sdris = []
        self.max_sdr = -10000000
        self.last_sdr = -10000000
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.last_path_ckpt = None
        self.filename_ckpt = config.FILENAME_CKPT
        self.save_hyperparameters()
 
        self.AE = self.pick_autoencoder(config.CHOSEN_AE.name, config)
        if self.chosen_model == cst.Autoencoders.VAE:
            self.ema = ExponentialMovingAverage(self.AE.parameters(), decay=0.99)
        else:
            self.ema = ExponentialMovingAverage(self.AE.codebooks.parameters(), decay=0.99)    

    
    def forward(self, mixture, is_train, batch_idx):
        recon, comm_loss = self.AE.forward(mixture, is_train, batch_idx)
        return recon, comm_loss
    
    def loss(self, x, recon, comm_loss):
        return self.AE.loss(x, recon, comm_loss)

    def forward_nade(self, x, is_train, batch_idx):
        losses = []
        if not is_train:
            p = 1.0
        else:
            p = self.p
        for i in range(self.num_steps):
            if i == 0:
                x_masked = self.masking(x, p)
                recon, comm_loss = self.AE.forward(x_masked, is_train, batch_idx)
                loss = self.AE.loss(x, recon, comm_loss)
                losses.append(loss)
            else:
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                    self.ema.update()
                    self.optimizer.zero_grad()
                recon = torch.cat([recon.detach(), x[:, -1:, :]], dim=-2)
                recon, comm_loss = self.AE.forward(recon, is_train, batch_idx)
                loss = self.AE.loss(x, recon, comm_loss)
                losses.append(loss)
        return recon, torch.stack(losses), loss

    def masking(self, x, p):
        # with probability p we mask the input
        mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        mask.requires_grad = False
        mask[torch.rand(x.shape[0], x.shape[1]) <= p] = 0.0
        mask[:, 4] = 1.0
        x = torch.einsum('b c l, b c -> b c l', x, mask)
        return x

    def training_step(self, x, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(x, dim=-2)
        mixture = rearrange(mixture, 'b t -> b 1 t')
        mixture.requires_grad = True
        #input = torch.cat([x, mixture], dim=-2)
        #input.requires_grad = True
        recon, comm_loss = self.forward(mixture, is_train=True, batch_idx=batch_idx)        
        loss = self.loss(x, recon, comm_loss)
        self.train_losses.append(loss.item())
        self.ema.update()
        return loss
    

    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')


    def on_validation_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch+1)
            wandb.log({'recon_time_term': sum(self.AE.recon_time_terms) / len(self.AE.recon_time_terms)}, step=self.current_epoch+1)
            wandb.log({'multi_spectral_recon_loss': sum(self.AE.multi_spectral_recon_losses) / len(self.AE.multi_spectral_recon_losses)}, step=self.current_epoch+1)
            if self.chosen_model == cst.Autoencoders.VQVAE or self.chosen_model == cst.Autoencoders.RQVAE:
                wandb.log({'commitment_loss': sum(self.AE.commitment_losses) / len(self.AE.commitment_losses)}, step=self.current_epoch+1)
        self.AE.recon_time_terms = []
        self.AE.multi_spectral_recon_losses = []
        self.AE.commitment_losses = []
        self.train_losses = []
        self.val_ema_sdr = []
        self.val_losses = []
        print(f'\ntrain loss on epoch {self.current_epoch} is {round(loss, 4)}')


    def validation_step(self, x, batch_idx):
        # from source to mixture
        mixture = torch.sum(x, dim=-2)
        mixture = rearrange(mixture, 'b t -> b 1 t')
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, comm_loss = self.forward(mixture, is_train=False, batch_idx=batch_idx)
            loss = self.loss(x, recon, comm_loss)
            self.val_ema_sdr.append(self.si_sdr(x, recon).item())
            self.val_losses.append(loss.item())
        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_sdr) / len(self.val_ema_sdr)
        self.log('val_ema_loss', self.val_loss)
        self.log('val_ema_sdr', loss_ema)
        # model checkpointing
        if loss_ema > self.max_sdr:
            # if the improvement is less than 0.05, we halve the learning rate        
            self.max_sdr = loss_ema
            self._model_checkpointing(loss_ema)
        # if there is no improvement we halve the learning rate
        if loss_ema - self.last_sdr < 0.05:
            self.optimizer.param_groups[0]["lr"] /= 2
        if self.IS_WANDB:
            wandb.log({'recon_time_term': sum(self.AE.recon_time_terms) / len(self.AE.recon_time_terms)}, step=self.current_epoch+1)
            wandb.log({'multi_spectral_recon_loss': sum(self.AE.multi_spectral_recon_losses) / len(self.AE.multi_spectral_recon_losses)}, step=self.current_epoch+1)
        self.AE.recon_time_terms = []
        self.last_sdr = loss_ema
        self.AE.multi_spectral_recon_losses = []
        print(f"\n val loss on epoch {self.current_epoch} is {round(self.val_loss, 4)}")
        print(f"\n val ema sdr on epoch {self.current_epoch} is {loss_ema}")



    def test_step(self, x, batch_idx):
        if batch_idx == 0:
            return 0
        flags = [is_silent(x[0][0]), is_silent(x[0][1]), is_silent(x[0][2]), is_silent(x[0][3])]
        result = sum(flags)
        if result > 2:
            return
        # from source to mixture
        mixture = torch.sum(x, dim=-2)
        mixture = rearrange(mixture, 'b t -> b 1 t')
        #input = torch.cat([x, mixture], dim=-2)
        recon, comm_loss = self.forward(mixture, is_train=False, batch_idx=batch_idx)
        loss = self.loss(x, recon, comm_loss)
        self.test_losses.append(loss.item())
        self.test_snrs.append(self.si_snr(x, recon).item())
        self.test_sdrs.append(self.si_sdr(x, recon).item())
        list_sdri = []
        self.list_sdri_guitar = []
        self.list_sdri_bass = []
        self.list_sdri_drums = []
        self.list_sdri_piano = []
        mixture = rearrange(mixture, 'b 1 t -> b t')
        for  i in range(len(cst.STEMS)):
            if i == 0:
                self.list_sdri_bass.append(self.si_sdr(x[:, i], recon[:, i]).item() - self.si_sdr(x[:, i], mixture).item())
            elif i == 1:
                self.list_sdri_drums.append(self.si_sdr(x[:, i], recon[:, i]).item() - self.si_sdr(x[:, i], mixture).item())
            elif i == 2:
                self.list_sdri_guitar.append(self.si_sdr(x[:, i], recon[:, i]).item() - self.si_sdr(x[:, i], mixture).item())
            elif i == 3:
                self.list_sdri_piano.append(self.si_sdr(x[:, i], recon[:, i]).item() - self.si_sdr(x[:, i], mixture).item())
            list_sdri.append(self.si_sdr(x[:, i], recon[:, i]).item() - self.si_sdr(x[:, i], mixture).item())
        self.test_sdris.append(sum(list_sdri) / len(list_sdri))

        if batch_idx % 1000 == 0 and self.IS_DEBUG:
            save_audio(recon[0][0], 'recon_bass'+str(batch_idx))
            save_audio(recon[0][1], 'recon_drums'+str(batch_idx))
            save_audio(recon[0][2], 'recon_guitar'+str(batch_idx))
            save_audio(recon[0][3], 'recon_piano'+str(batch_idx))
            save_audio(x[0][0], 'source_bass'+str(batch_idx))
            save_audio(x[0][1], 'source_drums'+str(batch_idx))
            save_audio(x[0][2], 'source_guitar'+str(batch_idx))
            save_audio(x[0][3], 'source_piano'+str(batch_idx))
            save_audio(mixture[0], 'source_mix'+str(batch_idx))
            save_audio((recon[0][0]+recon[0][1]+recon[0][2]+recon[0][3])//4, 'recon_mix'+str(batch_idx))
        return loss


    def on_test_end(self) -> None:
        print("len test losses: ", len(self.test_losses))
        test_loss = sum(self.test_losses) / len(self.test_losses)
        test_snr = sum(self.test_snrs) / len(self.test_snrs)
        test_sdr = sum(self.test_sdrs) / len(self.test_sdrs)
        test_sdri = sum(self.test_sdris) / len(self.test_sdris)
        test_sdri_bass = sum(self.list_sdri_bass) / len(self.list_sdri_bass)
        test_sdri_drums = sum(self.list_sdri_drums) / len(self.list_sdri_drums)
        test_sdri_guitar = sum(self.list_sdri_guitar) / len(self.list_sdri_guitar)
        test_sdri_piano = sum(self.list_sdri_piano) / len(self.list_sdri_piano)
        if self.IS_WANDB:
            wandb.log({'test_snr': test_snr}, step=self.current_epoch+1)
            wandb.log({'test_loss': test_loss}, step=self.current_epoch+1)
            wandb.log({'test_sdr': test_sdr}, step=self.current_epoch+1)
            wandb.log({'test_sdrsi': test_sdri}, step=self.current_epoch+1)
            #wandb.log({'test_ema_loss': loss_ema}, step=self.current_epoch+1)
        print(f"\n test loss on epoch {self.current_epoch} is {round(test_loss, 4)}")
        print(f"\n test snr on epoch {self.current_epoch} is {round(test_snr, 4)}")
        print(f"\n test sdr on epoch {self.current_epoch} is {round(test_sdr, 4)}")
        print(f"\n test sdri on epoch {self.current_epoch} is {round(test_sdri, 4)}")
        print(f"\n test sdr on epoch {self.current_epoch} is {round(test_sdr, 4)}")
        print(f"\n test sdri_bass on epoch {self.current_epoch} is {round(test_sdri_bass, 4)}")
        print(f"\n test sdri_drums on epoch {self.current_epoch} is {round(test_sdri_drums, 4)}")
        print(f"\n test sdri_guitar on epoch {self.current_epoch} is {round(test_sdri_guitar, 4)}")
        print(f"\n test sdri_piano on epoch {self.current_epoch} is {round(test_sdri_piano, 4)}")


    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer


    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_sdr", summary="max")
        wandb.define_metric("val_snr", summary="max")
        wandb.define_metric("val_sdr", summary="max")

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
        with self.ema.average_parameters():
            filename_ckpt = ("val_sdr=" + str(round(loss, 4)) +
                                "_epoch=" + str(self.current_epoch) +
                                "_" + self.filename_ckpt
                                )
            path_ckpt = cst.DIR_SAVED_MODEL + "/" + self.chosen_model.name + "/" + filename_ckpt
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






