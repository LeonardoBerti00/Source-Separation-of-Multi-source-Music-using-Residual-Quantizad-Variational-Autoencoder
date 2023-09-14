import torch
from einops import rearrange
import lightning as L
from repo.constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


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
        if config.HYPER_PARAMETERS[LearningHyperParameter.IS_ONED]:
            self.encoder = Encoder_CNN1D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                channel_size=config.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS],
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
                channel_size=config.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS],
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
                channel_size=config.HYPER_PARAMETERS[LearningHyperParameter.CHANNEL_SIZE],
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
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
            )
            self.decoder = Decoder_CNN2D(
                input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
                channel_size=config.HYPER_PARAMETERS[LearningHyperParameter.CHANNEL_SIZE],
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
                audio_srcs=config.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS],
            )
        self.codebook_length = config.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH]
        self.latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        self.audio_srcs = config.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS]
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.HYPER_PARAMETERS[LearningHyperParameter.IS_TRAINING]
        self.codebook = nn.Parameter(torch.Tensor(self.codebook_length, self.latent_dim))
        torch.nn.init.xavier_normal_(self.codebook, gain=torch.nn.init.calculate_gain("relu"))
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
    def forward(self, x):

        z_e = self.encoder(x)
        z_q = self.vector_quantization(z_e, self.codebook)

        recon = self.decoder(z_q)

        return recon, z_q, z_e

    def vector_quantization(self, z_e, codebook):

        dist = torch.cdist(z_e, codebook)
        _, min_idx = torch.min(dist, dim=2)
        z_q = torch.zeros(z_e.shape[0], z_e.shape[1], z_e.shape[2]).to(z_e.device)
        for i in range(self.batch_size):
            z_q[i] = torch.index_select(codebook, 0, min_idx[i])

        return z_q

    def training_step(self, x, batch_idx):
        recon, z_q, z_e = self.forward(x)
        loss = self.loss(x, recon, z_q, z_e)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, x, batch_idx):
        recon, z_q, z_e = self.forward(x)
        loss = self.loss(x, recon, z_q, z_e)
        self.log('val_loss', loss)
        return loss

    def test_step(self, x, batch_idx):
        recon, z_q, z_e = self.forward(x)
        loss = self.loss(x, recon, z_q, z_e)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer

    def loss(self, input, recon, z_q, z_e):
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = torch.mean(torch.sum((input - recon) ** 2, dim=1))

        return recon_term

class Encoder_CNN2D(nn.Module):
    def __init__(self,
                 input_size: int,
                 channel_size: int,
                 hidden1_channels: int,
                 hidden2_channels: int,
                 hidden3_channels: int,
                 hidden4_channels: int,
                 hidden5_channels: int,
                 kernel1_size: tuple,
                 kernel2_size: tuple,
                 kernel3_size: tuple,
                 kernel4_size: tuple,
                 kernel5_size: tuple,
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
        self.conv1 = nn.Conv2d(in_channels=channel_size, out_channels=hidden1_channels, kernel_size=kernel1_size, stride=kernel1_size, padding=(int(kernel1_size[0]/2), int(kernel1_size[1]/2)))
        self.conv2 = nn.Conv2d(in_channels=hidden1_channels, out_channels=hidden2_channels, kernel_size=kernel2_size, stride=kernel2_size, padding=(int(kernel2_size[0]/2), int(kernel2_size[1]/2)))
        self.conv3 = nn.Conv2d(in_channels=hidden2_channels, out_channels=hidden3_channels, kernel_size=kernel3_size, stride=kernel3_size, padding=(int(kernel3_size[0]/2), int(kernel3_size[1]/2)))
        self.conv4 = nn.Conv2d(in_channels=hidden3_channels, out_channels=hidden4_channels, kernel_size=kernel4_size, stride=kernel4_size, padding=(int(kernel4_size[0]/2), int(kernel4_size[1]/2)))
        self.conv5 = nn.Conv2d(in_channels=hidden4_channels, out_channels=hidden5_channels, kernel_size=kernel5_size, stride=kernel5_size, padding=(0, int(kernel5_size[1]/2)))
        self.lstm = nn.LSTM(input_size=hidden5_channels, hidden_size=latent_dim, num_layers=lstm_layers, batch_first=True)
        self.input_size = input_size

    def forward(self, x):
        x = rearrange(x, 'b h w -> b 1 h w')
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = rearrange(x, 'b c h w -> b w (c h)')
        print(x.shape)
        x, _ = self.lstm(x)
        print(x.shape)
        return x


class Decoder_CNN2D(nn.Module):
    def __init__(self,
                 input_size: int,
                 channel_size: int,
                 hidden1_channels: int,
                 hidden2_channels: int,
                 hidden3_channels: int,
                 hidden4_channels: int,
                 hidden5_channels: int,
                 kernel1_size: tuple,
                 kernel2_size: tuple,
                 kernel3_size: tuple,
                 kernel4_size: tuple,
                 kernel5_size: tuple,
                 latent_dim: int,
                 audio_srcs: int,
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
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden5_channels, num_layers=1, batch_first=True)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden5_channels, out_channels=hidden4_channels, kernel_size=kernel5_size, stride=kernel5_size, padding=(int(kernel5_size[0]/2)-2, int(kernel5_size[1]/2)))
        self.conv2 = nn.ConvTranspose2d(in_channels=hidden4_channels, out_channels=hidden3_channels, kernel_size=kernel4_size, stride=kernel4_size, padding=(int(kernel4_size[0]/2), int(kernel4_size[1]/2)-1))
        self.conv3 = nn.ConvTranspose2d(in_channels=hidden3_channels, out_channels=hidden2_channels, kernel_size=kernel3_size, stride=kernel3_size, padding=(int(kernel3_size[0]/2), int(kernel3_size[1]/2)-1))
        self.conv4 = nn.ConvTranspose2d(in_channels=hidden2_channels, out_channels=hidden1_channels, kernel_size=kernel2_size, stride=kernel2_size, padding=(int(kernel2_size[0]/2), int(kernel2_size[1]/2)-2))
        self.conv5 = nn.ConvTranspose2d(in_channels=hidden1_channels, out_channels=channel_size, kernel_size=kernel1_size, stride=kernel1_size, padding=(int(kernel1_size[0]/2), int(kernel1_size[1]/2)-2))
        self.hidden5_channels = hidden5_channels
        self.input_size = input_size
        self.channel_size = channel_size
        self.audio_srcs = audio_srcs

    def forward(self, x):
        print("decoder")
        print(x.shape)
        x, _ = self.lstm(x)
        print(x.shape)
        x = rearrange(x, 'b w c -> b c 1 w')
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.relu(self.conv4(x))
        print(x.shape)
        recon = F.relu(self.conv5(x))
        print(x.shape)
        return recon