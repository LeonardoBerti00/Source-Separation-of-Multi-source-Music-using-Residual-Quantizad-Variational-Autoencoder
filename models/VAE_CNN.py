import torch
from einops import rearrange
import lightning as L
from repo.constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
from models.Encoders import Encoder_CNN2D, Encoder_CNN1D
from models.Decoders import Decoder_CNN2D, Decoder_CNN1D
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from utils import compute_output_dim_conv, compute_output_dim_convtranspose


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
        self.latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        self.audio_srcs = config.AUDIO_SRCS
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.train_snr = []
        self.val_snr = []
        self.test_snr = []
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.IS_ONED = config.IS_ONED
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


        self.fc_mu = nn.Linear(in_features=self.latent_dim*emb_sample_len,
                               out_features=self.latent_dim)
        self.fc_log_var = nn.Linear(in_features=self.latent_dim*emb_sample_len,
                                    out_features=self.latent_dim)

        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim*emb_sample_len)

    def forward(self, x):

        x = self.encoder(x)
        x = rearrange(x, 'b c t -> b (c t)')
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        if self.training:
            var = torch.exp(0.5 * log_var)
            #print(torch.mean(var))
            #print(torch.mean(mean))
            normal_distribution = torch.distributions.Normal(loc=mean, scale=var)
            sampling = normal_distribution.rsample()    # rsample implements the reparametrization trick

        else:
            sampling = mean

        sampling = self.fc(sampling)
        sampling = rearrange(sampling, 'b (c w) -> b c w', c=self.latent_dim)
        recon = self.decoder(sampling)
        if not self.IS_ONED:
            # reshaping as the original input
            recon = rearrange(recon, 'b 1 h w -> b h w')
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
        self.train_snr.append(self.si_snr(x, recon))
        return loss

    def validation_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        self.val_snr.append(self.si_snr(x, recon))
        return loss

    def test_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.log('test_loss', loss)
        self.test_losses.append(loss)
        self.test_snr.append(self.si_snr(x, recon))
        return loss

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

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer



