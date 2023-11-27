import time

import torch
import wandb
from einops import rearrange
import lightning as L
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from config import Configuration
from constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
import constants as cst
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from models.Encoders import Encoder_CNN2D, Encoder_CNN1D
from models.Decoders import Decoder_CNN2D, Decoder_CNN1D
from utils.utils import compute_output_dim_convtranspose, compute_output_dim_conv


class VQVAE(L.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()

        self.IS_RESIDUAL = config.HYPER_PARAMETERS[LearningHyperParameter.IS_RESIDUAL]
        self.IS_WANDB = config.IS_WANDB
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.codebook_length = config.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH]
        self.latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.is_training = config.IS_TRAINING
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.beta = config.HYPER_PARAMETERS[LearningHyperParameter.BETA]
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.val_snr, self.test_snr = [], []
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.is_oned = config.IS_ONED
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)

        num_convs = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_CONVS]
        paddings = config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS]
        dilations = config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS]
        strides = config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES]
        init_sample_len = config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH]
        kernel_sizes = config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES]
        for i in range(num_convs):
            if i == 0:
                emb_sample_len = compute_output_dim_conv(input_dim=init_sample_len,
                                                         kernel_size=kernel_sizes[i][1],
                                                         padding=paddings[i][1],
                                                         dilation=dilations[i][1],
                                                         stride=strides[i][1])
            else:
                emb_sample_len = compute_output_dim_conv(input_dim=emb_sample_len,
                                                         kernel_size=kernel_sizes[i][1],
                                                         padding=paddings[i][1],
                                                         dilation=dilations[i][1],
                                                         stride=strides[i][1])
        self.codebook = nn.Embedding(self.codebook_length, self.latent_dim)
        self.codebook.weight.data.uniform_(-1 / self.codebook_length, 1 / self.codebook_length)

        if config.IS_ONED:
            self.encoder = Encoder_CNN1D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                audio_srcs=len(cst.STEMS),
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                dilations=dilations,
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                num_convs=num_convs,
                is_residual=self.IS_RESIDUAL,
            )
            self.decoder = Decoder_CNN1D(
                audio_srcs=len(cst.STEMS),
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                dilations=dilations,
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                batch_size=config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE],
                emb_sample_len=emb_sample_len,
                num_convs=num_convs,
                is_residual=self.IS_RESIDUAL,
                duration=config.HYPER_PARAMETERS[LearningHyperParameter.DURATION],
            )

        else:
            self.encoder = Encoder_CNN2D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                dilations=dilations,
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                num_convs=num_convs
            )
            self.decoder = Decoder_CNN2D(
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                dilations=dilations,
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                batch_size=config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE],
                emb_sample_len=emb_sample_len,
                num_convs=num_convs
            )

    def forward(self, x, is_train):
        #adding channel dimension
        x = rearrange(x, 'b l-> b 1 l')

        # if it is 1D we transform 4 x 22050 in 32 patches of size 32.
        z_e, encoding_indices = self.encode(x)

        recon, z_q = self.decode(z_e, encoding_indices, x.shape[0], is_train)

        return recon, z_q, z_e


    def encode(self, x):
        x = self.encoder(x)
        z_e = rearrange(x, 'b w c -> (b w) c').contiguous()
        dist = torch.cdist(z_e, self.codebook.weight)
        _, encoding_indices = torch.min(dist, dim=1)
        return z_e, encoding_indices

    def decode(self, z_e, encoding_indices, batch_size, is_train):
        encodings = F.one_hot(encoding_indices, self.codebook_length).float()
        z_q = torch.matmul(encodings, self.codebook.weight)

        if is_train:
            # straight-through gradient estimation: we attach z_q to the computational graph
            # to subsequently pass the gradient unaltered from the decoder to the encoder
            # whatever the gradients you get for the quantized variable z_q they will just be copy-pasted into z_e
            z_q = z_e + (z_q - z_e).detach()

        z_q_rearranged = rearrange(z_q, '(b l) c -> b l c', b=batch_size).contiguous()
        recon = self.decoder(z_q_rearranged)

        if not self.is_oned:
            # reshaping as the original input
            recon = rearrange(recon, 'b 1 h w -> b h w')
        return recon, z_q

    def loss(self, input, recon, z_q, z_e):
        # Commitment loss is the mse between the quantized latent vector and the encoder output
        q_latent_loss = F.mse_loss(z_e.detach(), z_q)      # we train the codebook
        e_latent_loss = F.mse_loss(z_e, z_q.detach())      # we train the encoder
        print(f"q_latent_loss: {q_latent_loss}")
        print(f"e_latent_loss: {e_latent_loss}")
        commitment_loss = q_latent_loss + self.beta * e_latent_loss
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = F.mse_loss(input, recon)
        print(f"recon term: {recon_term}")
        return recon_term + commitment_loss

    def training_step(self, x, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(x, dim=-2)
        mixture.requires_grad = True
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        recon, z_q, z_e = self.forward(mixture, is_train=True)
        batch_losses = self.loss(x, recon, z_q, z_e)
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        return batch_loss_mean

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch+1)
        print(f'train loss on epoch {self.current_epoch} is {loss}')

    def validation_step(self, x, batch_idx):
        # from source to mixture
        mixture = torch.sum(x, dim=-2)
        recon, z_q, z_e = self.forward(mixture, is_train=False)
        batch_losses = self.loss(x, recon, z_q, z_e)
        batch_loss_mean = torch.mean(batch_losses)
        self.val_losses.append(batch_loss_mean.item())
        self.val_snr.append(self.si_snr(x, recon))
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, z_q, z_e = self.forward(mixture, is_train=False)
            batch_ema_losses = self.loss(x, recon, z_q, z_e)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        return batch_loss_mean

    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', self.val_loss)
        if self.IS_WANDB:
            wandb.log({'val_ema_loss': loss_ema}, step=self.current_epoch)
        print(f"\n val loss on epoch {self.current_epoch} is {self.val_loss}")
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")

    def test_step(self, x, batch_idx):
        # from source to mixture
        mixture = torch.sum(x, dim=-2)
        recon, z_q, z_e = self.forward(mixture, is_train=False)
        batch_losses = self.loss(x, recon, z_q, z_e)
        batch_loss_mean = torch.mean(batch_losses)
        self.test_losses.append(batch_loss_mean.item())
        self.test_snr.append(self.si_snr(x, recon))
        '''
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[
            batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.test_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        '''
        # Testing: with EMA
        with self.ema.average_parameters():
            recon, z_q, z_e = self.forward(mixture, is_train=False)
            batch_ema_losses = self.loss(x, recon, z_q, z_e)
            ema_loss = torch.mean(batch_ema_losses)
            self.test_ema_losses.append(ema_loss.item())
            '''
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.test_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
            '''
        return batch_loss_mean

    def on_test_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        snr = sum(self.test_snr) / len(self.test_snr)
        if self.IS_WANDB:
            wandb.log({'snr': snr})
            wandb.log({'test_loss': loss})
            wandb.log({'test_ema_loss': loss_ema})
        print(f"\n test loss on epoch {self.current_epoch} is {loss}")
        print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")
        print(f"\n test snr on epoch {self.current_epoch} is {snr}")
        #numpy.save(cst.RECON_DIR + "/test_reconstructions.npy", self.test_reconstructions)
        #numpy.save(cst.RECON_DIR + "/test_ema_reconstructions.npy", self.test_ema_reconstructions)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")
        wandb.define_metric("test_loss", summary="min")
        wandb.define_metric("test_ema_loss", summary="min")

    def inference_time(self, x):
        t0 = time.time()
        _ = self.forward(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed




