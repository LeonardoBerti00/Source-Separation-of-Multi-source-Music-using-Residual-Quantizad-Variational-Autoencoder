import torch
from einops import rearrange
import lightning as L
from models.Encoders import Encoder_MLP
from models.Decoders import Decoder_MLP
from constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F


class VAE_MLP(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder_MLP(
            input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
            audio_srcs=config.AUDIO_SRCS,
            hidden_dims=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_MLP],
        )

        self.decoder = Decoder_MLP(
            input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
            audio_srcs=config.AUDIO_SRCS,
            hidden_dims=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_MLP],
        )

        latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        hidden_dims = config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_MLP]
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.fc_mu = nn.Linear(in_features=hidden_dims[-1],
                               out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=hidden_dims[-1],
                                    out_features=latent_dim)
        self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_dims[-1])

    def forward(self, x):

        x = self.encoder(x)
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        print(f"nan in mean: {torch.isnan(mean).any()}")
        print(f"nan in log_var: {torch.isnan(log_var).any()}")
        if self.training:
            var = torch.exp(0.5 * log_var)
            normal_distribution = torch.distributions.Normal(loc=mean, scale=var)
            sampling = normal_distribution.rsample()  # rsample implements the reparametrization trick

        else:
            sampling = mean

        sampling = self.fc(sampling)
        recon = self.decoder(sampling)

        return recon, mean, log_var

    def training_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
        self.log('val_loss', loss)
        return loss

    def test_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        loss = self.loss(x, recon, mean, log_var)
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

    def loss(self, input, recon, mean, log_var):
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = torch.mean(torch.sum((input - recon) ** 2, dim=1))

        # as second option of reconstruction loss, we can use the binary cross entropy loss, but in this case is better
        # to use the mse because the input is not binary, neither discrete but continuous
        # recon_term = F.binary_cross_entropy(recon, x, reduction='sum')

        # KL divergence is the difference between the distribution of the latent space and a normal distribution
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=1))

        return recon_term + kl_divergence





