import torch
from einops import rearrange
import lightning as L

from constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F

class VAE_MLP(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder_MLP(
            input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
            audio_srcs=config.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS],
            latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
            hidden1_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN1_MLP],
            hidden2_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN2_MLP],
            hidden3_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN3_MLP],
            hidden4_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN4_MLP]
            )

        self.decoder = Decoder_MLP(
            input_size=config.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH],
            audio_srcs=config.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS],
            latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
            hidden1_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN1_MLP],
            hidden2_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN2_MLP],
            hidden3_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN3_MLP],
            hidden4_dim=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN4_MLP]
            )

        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]

    def forward(self, x):

        mean, log_var = self.encoder(x)
        print(f"nan in mean: {torch.isnan(mean).any()}")
        print(f"nan in log_var: {torch.isnan(log_var).any()}")
        if self.training:
            var = torch.exp(0.5 * log_var)
            normal_distribution = torch.distributions.Normal(loc=mean, scale=var)
            sampling = normal_distribution.rsample()    # rsample implements the reparametrization trick

        else:
            sampling = mean

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
        #recon_term = F.binary_cross_entropy(recon, x, reduction='sum')

        # KL divergence is the difference between the distribution of the latent space and a normal distribution
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=1))

        return recon_term + kl_divergence



class Encoder_MLP(nn.Module):
    def __init__(self, input_size: int, audio_srcs: int, hidden1_dim: int, hidden2_dim: int, hidden3_dim: int, hidden4_dim: int, latent_dim: int):
        super().__init__()
        """
        This is a multilayer perceptron (MLP) used as encoder and decoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - audio_srcs: int, the number of channels in the input
        - output_size: int, the size of the output dimension
        - hidden1: int, the size of the first hidden layer
        - hidden2: int, the size of the second hidden layer
        - hidden3: int, the size of the third hidden layer
        """
        self.fc1 = nn.Linear(input_size * audio_srcs, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.fc_mu = nn.Linear(in_features=hidden4_dim,
                               out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=hidden4_dim,
                                    out_features=latent_dim)

    def forward(self, x):
        """
        This is the forward pass of the MLP.

        Parameters:
        - x: tensor of shape (batch_size, input_size, audio_srcs), the input data

        Returns:
        - mean: tensor of shape (batch_size, latent_dim), the mean of the latent distribution
        - log_var: tensor of shape (batch_size, latent_dim), the log variance of the latent distribution
        """

        # first we flatten the input, so that we can feed it to the linear layers, then we pass it through the linear layers
        x = rearrange(x, 'b d c -> b (d c)')
        print(f"nan in x: {torch.isnan(x).any()}")
        print(f"mean of x: {torch.mean(x)}")
        x = F.relu(self.fc1(x))
        print(f"nan in x: {torch.isnan(x).any()}")
        print(f"mean of x: {torch.mean(x)}")
        x = F.relu(self.fc2(x))
        print(f"nan in x: {torch.isnan(x).any()}")
        print(f"mean of x: {torch.mean(x)}")
        x = F.relu(self.fc3(x))
        print(f"nan in x: {torch.isnan(x).any()}")
        print(f"mean of x: {torch.mean(x)}")
        x = F.relu(self.fc4(x))
        print(f"nan in x: {torch.isnan(x).any()}")
        print(f"mean of x: {torch.mean(x)}")
        # we compute the mean and the log_var
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mean, log_var


class Decoder_MLP(nn.Module):
    def __init__(self, input_size: int, audio_srcs: int, latent_dim: int, hidden1_dim: int, hidden2_dim: int, hidden3_dim: int, hidden4_dim: int):
        super().__init__()
        """
        This is a multilayer perceptron (MLP) used as encoder and decoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - audio_srcs: int, the number of channels in the input
        - output_size: int, the size of the output dimension
        - hidden1: int, the size of the first hidden layer
        - hidden2: int, the size of the second hidden layer
        - hidden3: int, the size of the third hidden layer
        """
        self.fc1 = nn.Linear(latent_dim, hidden4_dim)
        self.fc2 = nn.Linear(hidden4_dim, hidden3_dim)
        self.fc3 = nn.Linear(hidden3_dim, hidden2_dim)
        self.fc4 = nn.Linear(hidden2_dim, hidden1_dim)
        self.fc5 = nn.Linear(hidden1_dim, input_size * audio_srcs)
        self.audio_srcs = audio_srcs

    def forward(self, x):
        """
        This is the forward pass of the MLP.

        Parameters:
        - x: tensor of shape (batch_size, input_size, audio_srcs), the input data

        Returns:
        - mean: tensor of shape (batch_size, latent_dim), the mean of the latent distribution
        - log_var: tensor of shape (batch_size, latent_dim), the log variance of the latent distribution
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        recon = rearrange(x, 'b (c d) -> b c d', c=self.audio_srcs)

        return recon