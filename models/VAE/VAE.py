import torch
import wandb
from einops import rearrange
import lightning as L
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage

from constants import LearningHyperParameter
from torch import nn
from models.Encoders import Encoder
from models.Decoders import Decoder
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from utils.utils import compute_output_dim_conv, compute_output_dim_convtranspose


class VAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        self.audio_srcs = config.AUDIO_SRCS
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
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
        init_sample_len = cst.SAMPLE_LENGTH
        kernel_sizes = config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES]
        num_convs = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_CONVS]
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)

        for i in range(num_convs):
            if i == 0:
                emb_sample_len = compute_output_dim_conv(input_dim=init_sample_len,
                                                    kernel_size=kernel_sizes[i],
                                                    padding=paddings[i],
                                                    dilation=dilations[i],
                                                    stride=strides[i])
            else:
                emb_sample_len = compute_output_dim_conv(input_dim=emb_sample_len,
                                                    kernel_size=kernel_sizes[i],
                                                    padding=paddings[i],
                                                    dilation=dilations[i],
                                                    stride=strides[i])

        self.encoder = Encoder(
            input_size=init_sample_len,
            audio_srcs=config.AUDIO_SRCS,
            hidden_channels=config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS],
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            dilations=dilations,
            latent_dim=config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM],
            lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
            num_convs=num_convs
        )
        self.decoder = Decoder(
            audio_srcs=config.AUDIO_SRCS,
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

        self.fc = nn.Linear(in_features=self.latent_dim*emb_sample_len,
                               out_features=2*self.latent_dim*emb_sample_len)

    def forward(self, x):

        x = self.encoder(x)
        x = rearrange(x, 'b c t -> b (c t)')
        mean, log_var = self.fc(x).chunk(2, dim=-1)
        if self.training:
            var = torch.exp(0.5 * log_var)
            #print(torch.mean(var))
            #print(torch.mean(mean))
            normal_distribution = torch.distributions.Normal(loc=mean, scale=var)
            sampling = normal_distribution.rsample()    # rsample implements the reparametrization trick
        else:
            sampling = mean

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
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        recon, mean, log_var = self.forward(x)
        batch_losses = self.loss(x, recon, mean, log_var)
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        return batch_loss_mean

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
        print(f'\ntrain loss on epoch {self.current_epoch} is {loss}')

    def validation_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        batch_losses = self.loss(x, recon, mean, log_var)
        batch_loss_mean = torch.mean(batch_losses)
        self.val_losses.append(batch_loss_mean.item())
        self.val_snr.append(self.si_snr(x, recon))
        # Validation: with EMA
        with self.ema.average_parameters():
            recon = self.forward(x)
            batch_ema_losses = self.loss(x, recon)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        self.val_snr.append(self.si_snr(x, recon))
        return batch_loss_mean

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', self.val_loss)
        if self.IS_WANDB:
            wandb.log({'val_ema_loss': loss_ema}, step=self.current_epoch)
        print(f"\n val loss on epoch {self.current_epoch} is {loss}")
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")

    def test_step(self, x, batch_idx):
        recon, mean, log_var = self.forward(x)
        batch_losses = self.loss(x, recon, mean, log_var)
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
            recon, commitment_loss = self.forward(x)
            batch_ema_losses = self.loss(x, recon, commitment_loss)
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

    def on_test_epoch_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        if self.IS_WANDB:
            wandb.log({'test_loss': loss})
            wandb.log({'test_ema_loss': loss_ema})
        print(f"\n test loss on epoch {self.current_epoch} is {loss}")
        print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")
        #numpy.save(cst.RECON_DIR + "/test_reconstructions.npy", self.test_reconstructions)
        #numpy.save(cst.RECON_DIR + "/test_ema_reconstructions.npy", self.test_ema_reconstructions)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer


