import torch
from einops import rearrange
import lightning as L
from repo.constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F


class VAE_CNN(L.LightningModule):
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
                audio_srcs=config.AUDIO_SRCS,
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
                audio_srcs=config.AUDIO_SRCS,
            )
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

    def forward(self, x):

        mean, log_var = self.encoder(x)
        if self.training:
            var = torch.exp(0.5 * log_var)
            #print(torch.mean(var))
            #print(torch.mean(mean))
            normal_distribution = torch.distributions.Normal(loc=mean, scale=var)
            sampling = normal_distribution.rsample()    # rsample implements the reparametrization trick

        else:
            sampling = mean

        recon = self.decoder(sampling)

        return recon, mean, log_var


    def loss(self, input, recon, mean, log_var):
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = torch.sqrt(torch.mean(torch.sum((input - recon) ** 2, dim=1)))

        # as second option of reconstruction loss, we can use the binary cross entropy loss, but in this case is better
        # to use the mse because the input is not binary, neither discrete but continuous
        #recon_term = F.binary_cross_entropy(recon, x, reduction='sum')

        # KL divergence is the difference between the distribution of the latent space and a normal distribution
        kl_divergence = torch.sqrt(-0.5 * torch.mean(torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=1)))

        return recon_term + kl_divergence



    def training_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.train_losses.append(loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        return loss

    def test_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.log('test_loss', loss)
        self.test_losses.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        print(f"\n train loss {loss}\n")

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        print(f"\n val loss {loss}")

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer




class Encoder_CNN1D(nn.Module):
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
                 padding: int,
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
        self.conv1 = nn.Conv1d(in_channels=channel_size, out_channels=hidden1_channels, kernel_size=kernel1_size[1], stride=kernel1_size[1], padding=padding)
        self.conv2 = nn.Conv1d(in_channels=hidden1_channels, out_channels=hidden2_channels, kernel_size=kernel2_size[1], stride=kernel2_size[1], padding=padding)
        self.conv3 = nn.Conv1d(in_channels=hidden2_channels, out_channels=hidden3_channels, kernel_size=kernel3_size[1], stride=kernel3_size[1], padding=padding)
        self.conv4 = nn.Conv1d(in_channels=hidden3_channels, out_channels=hidden4_channels, kernel_size=kernel4_size[1], stride=kernel4_size[1], padding=padding)
        self.conv5 = nn.Conv1d(in_channels=hidden4_channels, out_channels=hidden5_channels, kernel_size=kernel5_size[1], stride=kernel5_size[1], padding=padding)

        self.fc_mu = nn.Linear(in_features=hidden5_channels*int(input_size/(kernel1_size[1]*kernel2_size[1]*kernel3_size[1]*kernel4_size[1]*kernel5_size[1])+1),
                               out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=hidden5_channels*int(input_size/(kernel1_size[1]*kernel2_size[1]*kernel3_size[1]*kernel4_size[1]*kernel5_size[1])+1),
                                    out_features=latent_dim)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


class Decoder_CNN1D(nn.Module):
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
                 padding: int,
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
        self.out_features = hidden5_channels*int(input_size/(kernel1_size[1]*kernel2_size[1]*kernel3_size[1]*kernel4_size[1]*kernel5_size[1])+1)
        self.fc = nn.Linear(in_features=latent_dim, out_features=self.out_features)
        self.conv1 = nn.ConvTranspose1d(in_channels=hidden5_channels, out_channels=hidden4_channels, kernel_size=kernel5_size[1], stride=kernel5_size[1], padding=padding)
        self.conv2 = nn.ConvTranspose1d(in_channels=hidden4_channels, out_channels=hidden3_channels, kernel_size=kernel4_size[1], stride=kernel4_size[1], padding=padding-1)
        self.conv3 = nn.ConvTranspose1d(in_channels=hidden3_channels, out_channels=hidden2_channels, kernel_size=kernel3_size[1], stride=kernel3_size[1], padding=padding-1)
        self.conv4 = nn.ConvTranspose1d(in_channels=hidden2_channels, out_channels=hidden1_channels, kernel_size=kernel2_size[1], stride=kernel2_size[1], padding=padding-1)
        self.conv5 = nn.ConvTranspose1d(in_channels=hidden1_channels, out_channels=channel_size, kernel_size=kernel1_size[1], stride=kernel1_size[1], padding=padding-1)
        self.hidden5_channels = hidden5_channels
        self.input_size = input_size
        self.kernel1_size = kernel1_size
        self.kernel2_size = kernel2_size
        self.kernel3_size = kernel3_size
        self.kernel4_size = kernel4_size
        self.kernel5_size = kernel5_size

    def forward(self, x):
        x = self.fc(x)
        x = rearrange(x, 'b (h n) -> b h n', h=self.hidden5_channels)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


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
        self.conv1 = nn.Conv2d(in_channels=channel_size, out_channels=hidden1_channels, kernel_size=kernel1_size, stride=kernel1_size, padding=(int(kernel1_size[0]/2), int(kernel1_size[1]/2)))
        self.conv2 = nn.Conv2d(in_channels=hidden1_channels, out_channels=hidden2_channels, kernel_size=kernel2_size, stride=kernel2_size, padding=(int(kernel2_size[0]/2), int(kernel2_size[1]/2)))
        self.conv3 = nn.Conv2d(in_channels=hidden2_channels, out_channels=hidden3_channels, kernel_size=kernel3_size, stride=kernel3_size, padding=(int(kernel3_size[0]/2), int(kernel3_size[1]/2)))
        self.conv4 = nn.Conv2d(in_channels=hidden3_channels, out_channels=hidden4_channels, kernel_size=kernel4_size, stride=kernel4_size, padding=(int(kernel4_size[0]/2), int(kernel4_size[1]/2)))
        self.conv5 = nn.Conv2d(in_channels=hidden4_channels, out_channels=hidden5_channels, kernel_size=kernel5_size, stride=kernel5_size, padding=(int(kernel5_size[0]/2), int(kernel5_size[1]/2)))

        self.dividend = int((kernel1_size[1]*kernel2_size[1]*kernel3_size[1]*kernel4_size[1]*kernel5_size[1]))
        self.in_features = hidden5_channels * (audio_srcs+int((input_size*audio_srcs)/self.dividend))

        self.fc_mu = nn.Linear(in_features=self.in_features,
                               out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=self.in_features,
                                    out_features=latent_dim)
        self.input_size = input_size

    def forward(self, x):
        x = rearrange(x, 'b h w -> b 1 h w')

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = rearrange(x, 'b c h w -> b (c h w)')

        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


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
        self.dividend = int((kernel1_size[1] * kernel2_size[1] * kernel3_size[1] * kernel4_size[1] * kernel5_size[1]))
        self.out_features = hidden5_channels * (audio_srcs + int((input_size * audio_srcs) / self.dividend))

        self.fc = nn.Linear(in_features=latent_dim,
                               out_features=self.out_features)
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden5_channels, out_channels=hidden4_channels, kernel_size=kernel5_size, stride=kernel5_size, padding=(int(kernel5_size[0]/2), int(kernel5_size[1]/2)))
        self.conv2 = nn.ConvTranspose2d(in_channels=hidden4_channels, out_channels=hidden3_channels, kernel_size=kernel4_size, stride=kernel4_size, padding=(int(kernel4_size[0]/2), int(kernel4_size[1]/2)-1))
        self.conv3 = nn.ConvTranspose2d(in_channels=hidden3_channels, out_channels=hidden2_channels, kernel_size=kernel3_size, stride=kernel3_size, padding=(int(kernel3_size[0]/2), int(kernel3_size[1]/2)-1))
        self.conv4 = nn.ConvTranspose2d(in_channels=hidden2_channels, out_channels=hidden1_channels, kernel_size=kernel2_size, stride=kernel2_size, padding=(int(kernel2_size[0]/2), int(kernel2_size[1]/2)-2))
        self.conv5 = nn.ConvTranspose2d(in_channels=hidden1_channels, out_channels=channel_size, kernel_size=kernel1_size, stride=kernel1_size, padding=(int(kernel1_size[0]/2), int(kernel1_size[1]/2)-2))
        self.hidden5_channels = hidden5_channels
        self.input_size = input_size
        self.channel_size = channel_size
        self.audio_srcs = audio_srcs

    def forward(self, x):
        x = self.fc(x)

        x = rearrange(x, 'b (c h w) -> b c h w', c=self.hidden5_channels, h=self.audio_srcs)

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        recon = F.relu(self.conv5(x))

        return recon
