import torch
from einops import rearrange
import lightning as L
from repo.constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import numpy as np

def compute_output_dim(input_dim, kernel_size, padding, dilation, stride):
    return ((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

def compute_output_dim_convtranspose(input_dim, kernel_size, padding, dilation, stride):
    return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

class VQVAE_CNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder and decoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - channel_size: int, the number of channels in the input
        - hidden1: int, the size of the first hidden layer
        - hidden2: int, the size of the second hidden layer
        - hidden3: int, the size of the third hidden layer
        - hidden4: int, the size of the fourth hidden layer
        - kernel_size: int, the size of the kernel
        - latent_dim: int, the size of the latent dimension
        """

        self.codebook_length = config.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH]
        self.latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        self.audio_srcs = config.AUDIO_SRCS
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.codebook = nn.Parameter(torch.Tensor(self.codebook_length, self.latent_dim))
        torch.nn.init.uniform_(self.codebook, -1 / self.codebook_length, 1 / self.codebook_length)
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.commitment_cost = config.HYPER_PARAMETERS[LearningHyperParameter.COMMITMENT_COST]
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        paddings = config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS]
        dilations = config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS]
        strides = config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES]
        init_sample_len = config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH]
        kernel_sizes = config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES]
        for i in range(len(kernel_sizes)):
            if i == 0:
                emb_sample_len = compute_output_dim(input_dim=init_sample_len,
                                                    kernel_size=kernel_sizes[i][1],
                                                    padding=paddings[i][1],
                                                    dilation=dilations[i][1],
                                                    stride=strides[i][1])
            else:
                emb_sample_len = compute_output_dim(input_dim=emb_sample_len,
                                                    kernel_size=kernel_sizes[i][1],
                                                    padding=paddings[i][1],
                                                    dilation=dilations[i][1],
                                                    stride=strides[i][1])
            print(emb_sample_len)
        if config.IS_ONED:
            self.encoder = Encoder_CNN1D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                channel_size=config.AUDIO_SRCS,
                hidden1_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN1_CHANNELS_CNN],
                hidden2_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN2_CHANNELS_CNN],
                hidden3_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN3_CHANNELS_CNN],
                hidden4_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN4_CHANNELS_CNN],
                hidden5_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN5_CHANNELS_CNN],
                kernel1_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL1_SIZE],
                kernel2_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL2_SIZE],
                kernel3_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL3_SIZE],
                kernel4_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL4_SIZE],
                kernel5_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL5_SIZE],
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                padding=config.HYPER_PARAMETERS[LearningHyperParameter.PADDING],
                )
            self.decoder = Decoder_CNN1D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                channel_size=config.AUDIO_SRCS,
                hidden1_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN1_CHANNELS_CNN],
                hidden2_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN2_CHANNELS_CNN],
                hidden3_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN3_CHANNELS_CNN],
                hidden4_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN4_CHANNELS_CNN],
                hidden5_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN5_CHANNELS_CNN],
                kernel1_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL1_SIZE],
                kernel2_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL2_SIZE],
                kernel3_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL3_SIZE],
                kernel4_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL4_SIZE],
                kernel5_size=config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL5_SIZE],
                latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
                padding=config.HYPER_PARAMETERS[LearningHyperParameter.PADDING],
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
                final_audio_len=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                emb_sample_len=emb_sample_len,
            )



    def forward(self, x):

        z_e = self.encoder(x)
        z_q = self.vector_quantization(z_e)

        # Commitment loss is the mse between the quantized latent vector and the encoder output
        q_latent_loss = F.mse_loss(z_e.detach(), z_q)
        e_latent_loss = F.mse_loss(z_e, z_q.detach())
        commitment_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # From decoder to encoder
        z_q = z_e + (z_q - z_e).detach()
        z_q = rearrange(z_q, '(b l) c -> b l c', b=x.shape[0])
        recon = self.decoder(z_q)

        return recon, commitment_loss

    def vector_quantization(self, z_e):

        dist = torch.cdist(z_e, self.codebook)
        _, encoding_indices = torch.min(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_length).float()
        z_q = torch.matmul(encodings, self.codebook.data)

        return z_q

    def training_step(self, x, batch_idx):
        recon, commitment_loss = self.forward(x)
        loss = self.loss(x, recon, commitment_loss)
        self.log('train_loss', loss)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, x, batch_idx):
        recon, commitment_loss = self.forward(x)
        loss = self.loss(x, recon, commitment_loss)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        return loss

    def test_step(self, x, batch_idx):
        recon, commitment_loss = self.forward(x)
        loss = self.loss(x, recon, commitment_loss)
        self.log('test_loss', loss)
        self.test_losses.append(loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer

    def loss(self, input, recon, commitment_loss):

        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = F.mse_loss(input, recon)

        return recon_term + commitment_loss

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        print(f"\n train loss {loss}\n")

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        print(f"\n val loss {loss}")


class Encoder_CNN2D(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 latent_dim: int,
                 lstm_layers: int,
                 ):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder and decoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - channel_size: int, the number of channels in the input
        - hidden1: int, the size of the first hidden layer
        - hidden2: int, the size of the second hidden layer
        - hidden3: int, the size of the third hidden layer
        - hidden4: int, the size of the fourth hidden layer
        - kernel_size: int, the size of the kernel
        - latent_dim: int, the size of the latent dimension
        """
        modules = []
        for i in range(len(kernel_sizes)):
            # Last layer merges the four audio sources
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels[i],
                          out_channels=hidden_channels[i+1],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          dilation=dilations[i],
                          padding=paddings[i],
                          ),
                nn.LeakyReLU(negative_slope=0.1),
                nn.BatchNorm2d(hidden_channels[i+1]))
            )

        self.LSTM = nn.LSTM(input_size=hidden_channels[-1], hidden_size=latent_dim, num_layers=lstm_layers, batch_first=True)
        self.Convs = nn.Sequential(*modules)
        self.input_size = input_size


    def forward(self, x):
        x = rearrange(x, 'b h w -> b 1 h w')
        print(x.shape)
        x = self.Convs(x)

        x = rearrange(x, 'b c h w -> b w (c h)')
        print(x.shape)
        x, _ = self.LSTM(x)
        print(x.shape)
        # rearrange motivated by VQ-VAE, in which they used channel dimension as latent dimension
        x = rearrange(x, 'b w c -> (b w) c')
        return x


class Decoder_CNN2D(nn.Module):
    def __init__(self,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 latent_dim: int,
                 lstm_layers: int,
                 batch_size: int,
                 final_audio_len: int,
                 emb_sample_len: int,
                 ):
        super().__init__()

        """
        Parameters:
        - hidden_channels: list, the number of channels in the hidden layers
        - kernel_sizes: list, the size of the kernels
        - latent_dim: int, the size of the latent dimension
        - lstm_layers: int, the number of lstm layers
        - batch_size: int, the size of the batch
        """

        self.LSTM = nn.LSTM(input_size=latent_dim, hidden_size=hidden_channels[-1], num_layers=lstm_layers, batch_first=True)
        #print(latent_dim)
        modules = []
        for i in range(1, len(kernel_sizes)+1):
            if i==1:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=emb_sample_len,
                                                                kernel_size=kernel_sizes[-i][1],
                                                                stride=strides[-i][1],
                                                                dilation=dilations[-i][1],
                                                                padding=paddings[-i][1]
                                                                )
            else:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=after_conv_sample_len,
                                                                         kernel_size=kernel_sizes[-i][1],
                                                                         stride=strides[-i][1],
                                                                         dilation=dilations[-i][1],
                                                                         padding=paddings[-i][1]
                                                                         )
            modules.append(nn.Sequential(
                #nn.ZeroPad2d((kernel_sizes[-i][1]//2, kernel_sizes[-i][1]//2, 0, 0)),
                nn.ConvTranspose2d(in_channels=hidden_channels[-i],
                                   out_channels=hidden_channels[-i-1],
                                   kernel_size=kernel_sizes[-i],
                                   stride=strides[-i],
                                   dilation=dilations[-i],
                                   padding=paddings[-i]
                                   ),
                nn.LeakyReLU(negative_slope=0.1))
            )
            print(after_conv_sample_len)

        self.fc = nn.Sequential(
            nn.Linear(in_features=after_conv_sample_len, out_features=final_audio_len),
        )
        self.Convs = nn.Sequential(*modules)
        self.batch_size = batch_size

    def forward(self, x):

        print("decoder")
        print(x.shape)
        x, _ = self.LSTM(x)
        print(x.shape)
        x = rearrange(x, 'b w c -> b c 1 w')
        print(x.shape)
        x = self.Convs(x)
        print(x.shape)
        #recon = self.fc(x)
        #print(recon.shape)
        #recon = rearrange(recon, 'b 1 h w -> b h w')
        return x