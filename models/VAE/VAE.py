import torch
from einops import rearrange
import lightning as L

from constants import LearningHyperParameter
from torch import nn
import torchaudio.transforms as T
from torch.linalg import vector_norm
from torch.nn import functional as F
from models.Encoders import Encoder
from models.Decoders import Decoder
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from utils.utils import compute_final_output_dim, compute_output_dim_conv, compute_output_dim_convtranspose
import constants as cst

class VAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.recon_time_terms = []
        self.multi_spectral_recon_losses = []
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.multi_spectral_recon_loss_weight = config.HYPER_PARAMETERS[LearningHyperParameter.MULTI_SPECTRAL_RECON_LOSS_WEIGHT]
        init_sample_len = cst.SAMPLE_LENGTH
        conv_setup = config.HYPER_PARAMETERS[LearningHyperParameter.CONV_SETUP]
        paddings = config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS][conv_setup]
        dilations = config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS][conv_setup]
        strides = config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES][conv_setup]
        kernel_sizes = config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES][conv_setup]
        num_convs = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_CONVS]
        hidden_channels = config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS][conv_setup]
        self.latent_dim = hidden_channels[-1]

        self.emb_sample_len = compute_final_output_dim(
            input_dim = init_sample_len,
            kernel_sizes = kernel_sizes,
            paddings = paddings,
            dilations = dilations,
            strides = strides,
            num_convs = num_convs
        )

        self.encoder = Encoder(
                input_size=init_sample_len,
                hidden_channels=hidden_channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                dilations=dilations,
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                num_convs=num_convs,
                dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT],
            )
        self.decoder = Decoder(
                audio_srcs=len(cst.STEMS),
                hidden_channels=hidden_channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                dilations=dilations,
                lstm_layers=config.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS],
                emb_sample_len=self.emb_sample_len,
                num_convs=num_convs,
                dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT],
            )

        self.fc = nn.Linear(in_features=self.latent_dim*int(self.emb_sample_len),
                               out_features=2*self.latent_dim*int(self.emb_sample_len))
        # initiliazation to compute the multi spectogram loss
        # a list of powers of two for the window sizes of each MelSpectrogram transform
        multi_spectral_window_powers = cst.MULTI_SPECTRAL_WINDOW_POWERS 
        # hyperparameter that controls the weight of the multi spectogram loss in the total loss function
        self.multi_spectral_recon_loss_weight = config.HYPER_PARAMETERS[LearningHyperParameter.MULTI_SPECTRAL_RECON_LOSS_WEIGHT] 
        num_transforms = len(multi_spectral_window_powers) 
        multi_spectral_n_ffts = ((cst.MULTI_SPECTRAL_N_FFTS,) * num_transforms)
        multi_spectral_n_mels = ((cst.MULTI_SPECTRAL_N_MELS,) * num_transforms) 
        self.mel_spec_transforms = []
        self.mel_spec_recon_alphas = []
        for powers, n_fft, n_mels in zip(multi_spectral_window_powers, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers 
            # calculate the alpha value for the current transform, which is used to scale the output of the MelSpectrogram
            alpha = (win_length / 2) ** 0.5 
            # calculate the number of frequency bins for the current transform
            calculated_n_fft = max(n_fft, win_length) 
            # create a MelSpectrogram transform object from the PyTorch library, using the calculated parameters and some constants
            melspec_transform = T.MelSpectrogram(
                sample_rate = cst.SAMPLE_RATE,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
            ).to(cst.DEVICE)

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)


    def forward(self, x, is_train, batch_idx):
        x = self.encoder(x)
        x = rearrange(x, 'b c l -> b (c l)')
        mean, log_var = self.fc(x).chunk(2, dim=-1)
        if self.training:
            var = torch.exp(0.5 * log_var)
            normal_distribution = torch.distributions.Normal(loc=mean, scale=var)
            sampling = normal_distribution.rsample()    # rsample implements the reparametrization trick
        else:
            sampling = mean
        sampling = rearrange(sampling, 'b (c l) -> b l c', c=self.latent_dim)
        recon = self.decoder(sampling)
        return recon, (mean, log_var)


    def loss(self, input, recon, params):
        mean, log_var = params
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_term = F.mse_loss(recon, input)
        # as second option of reconstruction loss, we can use the binary cross entropy loss, but in this case is better
        # to use the mse because the input is not binary, neither discrete but continuous
        #recon_term = F.binary_cross_entropy(recon, x, reduction='sum')

        # KL divergence is the difference between the distribution of the latent space and a normal distribution
        kl_divergence = torch.sqrt(-0.5 * torch.mean(torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=1)))
        
        multi_spectral_recon_loss = 0
        for src in range(len(cst.STEMS)):
            for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
                orig_mel, recon_mel = map(mel_transform, (input[:, src, :], recon[:, src, :]))
                log_orig_mel, log_recon_mel = torch.log(orig_mel), torch.log(recon_mel)

                l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim = -2).mean()
                l2_log_mel_loss = vector_norm(log_orig_mel - log_recon_mel, dim = -2).mean()

                multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss
        self.recon_time_terms.append(recon_term.item())
        self.multi_spectral_recon_losses.append(multi_spectral_recon_loss.item())
        #self.commitment_losses.append(commitment_loss.item())
        return recon_term + kl_divergence + self.multi_spectral_recon_loss_weight*multi_spectral_recon_loss


