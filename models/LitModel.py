import numpy
from config import Configuration
import torch
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time
from torch_ema import ExponentialMovingAverage
import wandb
from utils.utils_models import pick_model, pick_autoencoder


class LitModel(L.LightningModule):

    def __init__(self, config: Configuration, val_num_steps: int, test_num_steps: int, trainer: L.Trainer):
        super().__init__()
        """
        This is the skeleton of the diffusion models.
        """
        self.trainer = trainer
        self.IS_WANDB = config.IS_WANDB
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.training = config.IS_TRAINING
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.val_num_batches = int(val_num_steps / self.batch_size) + 1
        self.test_num_batches = int(test_num_steps / self.batch_size) + 1
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.test_reconstructions = numpy.zeros((test_num_steps, len(cst.STEMS)))
        self.test_ema_reconstructions = numpy.zeros((test_num_steps, len(cst.STEMS)))
        self.model = pick_model(config, config.CHOSEN_MODEL.name)
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.AE = pick_autoencoder(config, config.CHOSEN_AE.name)

    def forward(self, x, is_train):

        return self.NN(x, is_train)

    def loss(self, real, recon, **kwargs):
        return self.NN.loss(real, recon, **kwargs)

    def training_step(self, input, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        output, reverse_context = self.forward(input, is_train=True)
        reverse_context.update({'is_train': True})
        batch_losses = self.loss(input, output, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        return batch_loss_mean

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
        print(f'train loss on epoch {self.current_epoch} is {loss}')

    def validation_step(self, input, batch_idx):
        recon, reverse_context = self.forward(input, is_train=False)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(input, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.val_losses.append(batch_loss_mean.item())
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(input, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(input, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        return batch_loss_mean

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', self.val_loss)
        if self.IS_WANDB:
            wandb.log({'val_ema_loss': loss_ema}, step=self.current_epoch)
        print(f"\n val loss on epoch {self.current_epoch} is {loss}")
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")

    def test_step(self, input, batch_idx):
        recon, reverse_context = self.forward(input, is_train=False)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(input, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.test_losses.append(batch_loss_mean.item())
        recon = self._to_original_dim(recon)
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[
            batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.test_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        # Testing: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(input, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(input, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.test_ema_losses.append(ema_loss.item())
            recon = self._to_original_dim(recon)
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.test_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        return batch_loss_mean

    def on_test_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        if self.IS_WANDB:
            wandb.log({'test_loss': loss})
            wandb.log({'test_ema_loss': loss_ema})
        print(f"\n test loss on epoch {self.current_epoch} is {loss}")
        print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")
        numpy.save(cst.RECON_DIR + "/test_reconstructions.npy", self.test_reconstructions)
        numpy.save(cst.RECON_DIR + "/test_ema_reconstructions.npy", self.test_ema_reconstructions)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def inference_time(self, x):
        t0 = time.time()
        _ = self.forward(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")
        wandb.define_metric("test_loss", summary="min")
        wandb.define_metric("test_ema_loss", summary="min")