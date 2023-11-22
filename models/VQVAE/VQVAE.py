import torch
from einops import rearrange
import lightning as L
from torch_ema import ExponentialMovingAverage
from config import Configuration
from constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from models.Encoders import Encoder_CNN2D, Encoder_CNN1D
from models.Decoders import Decoder_CNN2D, Decoder_CNN1D
from utils.utils import compute_output_dim_convtranspose, compute_output_dim_conv


class VQVAE(L.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()

        self.codebook_length = config.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH]
        self.latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        self.audio_srcs = config.AUDIO_SRCS
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.is_training = config.IS_TRAINING
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.beta = config.HYPER_PARAMETERS[LearningHyperParameter.BETA]
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.val_snr, self.test_snr = [], []
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.is_oned = config.IS_ONED
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)

        paddings = config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS]
        dilations = config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS]
        strides = config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES]
        init_sample_len = config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH]
        kernel_sizes = config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES]
        for i in range(len(kernel_sizes)):
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

        self.codebook = nn.Parameter(torch.Tensor(self.codebook_length, self.latent_dim))
        torch.nn.init.uniform_(self.codebook, -1 / self.codebook_length, 1 / self.codebook_length)

        if config.IS_ONED:
            self.encoder = Encoder_CNN1D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                audio_srcs=config.AUDIO_SRCS,
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES],
                strides=config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES],
                paddings=config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS],
                dilations=config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS],
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
            )
            self.decoder = Decoder_CNN1D(
                audio_srcs=config.AUDIO_SRCS,
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES],
                strides=config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES],
                paddings=config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS],
                dilations=config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS],
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                batch_size=config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE],
                emb_sample_len=emb_sample_len,
            )

        else:
            self.encoder = Encoder_CNN2D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES],
                strides=config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES],
                paddings=config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS],
                dilations=config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS],
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
            )
            self.decoder = Decoder_CNN2D(
                hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
                kernel_sizes=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES],
                strides=config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES],
                paddings=config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS],
                dilations=config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS],
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                batch_size=config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE],
                emb_sample_len=emb_sample_len,
            )

    def forward(self, x):
        #if it is 1D we transform 4 x 22050 in 32 patches of size 32.
        z_e, z_q = self.encode(x)
        z_q = z_e + (z_q - z_e).detach()
        z_q = rearrange(z_q, '(b l) c -> b l c', b=x.shape[0])
        recon = self.decoder(z_q)

        if not self.is_oned:
            # reshaping as the original input
            recon = rearrange(recon, 'b 1 h w -> b h w')
        return recon, z_q, z_e

    def encode(self, x):
        x = self.encoder(x)
        z_e = rearrange(x, 'b w c -> (b w) c')
        z_q = self.vector_quantization(z_e)
        return z_e, z_q

    def vector_quantization(self, z_e):
        dist = torch.cdist(z_e, self.codebook)
        _, encoding_indices = torch.min(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_length).float()
        z_q = torch.matmul(encodings, self.codebook.data)
        return z_q

    def loss(self, input, recon, z_q, z_e):
        # Commitment loss is the mse between the quantized latent vector and the encoder output
        q_latent_loss = F.mse_loss(z_e.detach(), z_q)      # we train the codebook
        e_latent_loss = F.mse_loss(z_e, z_q.detach())      # we train the encoder
        commitment_loss = q_latent_loss + self.beta * e_latent_loss
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = F.mse_loss(input, recon)
        return recon_term + commitment_loss

    def training_step(self, x, batch_idx):
        recon, z_q, z_e = self.forward(x)
        loss = self.loss(x, recon, z_q, z_e)
        self.log('train_loss', loss)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, x, batch_idx):
        recon, commitment_loss = self.forward(x)
        loss = self.loss(x, recon, commitment_loss)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        self.val_snr.append(self.si_snr(x, recon))
        return loss

    def test_step(self, x, batch_idx):
        recon, commitment_loss = self.forward(x)
        loss = self.loss(x, recon, commitment_loss)
        self.log('test_loss', loss)
        self.test_losses.append(loss)
        self.test_snr.append(self.si_snr(x, recon))
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer



    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        print(f"\n train loss {loss}\n")

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        print(f"\n val loss {loss}")
        print(f"\n val snr {sum(self.val_snr) / len(self.val_snr)}\n")

    def on_test_epoch_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        print(f"\n test loss {loss}")
        print(f"\n test snr {sum(self.test_snr) / len(self.test_snr)}\n")
