import random
import numpy as np
import torch
from einops import rearrange
from torch_ema import ExponentialMovingAverage
from config import Configuration
from constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torch.linalg import vector_norm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from scipy.cluster.vq import kmeans
import torchaudio.transforms as T

import constants as cst
from models.Encoders import Encoder
from models.Decoders import Decoder
from utils.utils import compute_final_output_dim
from vector_quantize_pytorch import ResidualVQ

class RQVAE(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()

        self.init_kmeans = config.HYPER_PARAMETERS[LearningHyperParameter.INIT_KMEANS]
        self.num_quantizers = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_QUANTIZERS]
        self.recon_loss_weight = config.HYPER_PARAMETERS[LearningHyperParameter.RECON_LOSS_WEIGHT]
        self.IS_DEBUG = config.IS_DEBUG
        self.codebook_length = config.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH]
        self.shared_codebook = config.HYPER_PARAMETERS[LearningHyperParameter.SHARED_CODEBOOK]
        self.is_training = config.IS_TRAINING
        self.recon_time_terms = []
        self.multi_spectral_recon_losses = []
        self.commitment_losses = []
        self.sdr_losses = []
        self.si_snr = ScaleInvariantSignalNoiseRatio()

        num_convs = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_CONVS]
        init_sample_len = cst.SAMPLE_LENGTH
        conv_setup = config.HYPER_PARAMETERS[LearningHyperParameter.CONV_SETUP]
        paddings = config.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS][conv_setup]
        dilations = config.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS][conv_setup]
        strides = config.HYPER_PARAMETERS[LearningHyperParameter.STRIDES][conv_setup]
        kernel_sizes = config.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES][conv_setup]
        hidden_channels = config.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS][conv_setup]

        self.emb_sample_len = compute_final_output_dim(
            input_dim = init_sample_len,
            kernel_sizes = kernel_sizes,
            paddings = paddings,
            dilations = dilations,
            strides = strides,
            num_convs = num_convs
        )
        self.latent_dim = hidden_channels[-1]
  
        if self.shared_codebook:
            self.codebooks = nn.ModuleList([nn.Embedding(self.codebook_length, self.latent_dim, device=cst.DEVICE)])
            self.codebooks.weight.data.uniform_(-1 / self.codebook_length, 1 / self.codebook_length)
            self.codebooks.weight.requires_grad = True
            self.codebooks_freq = torch.zeros(1, self.codebook_length, device=cst.DEVICE)
        else:
            self.codebooks = nn.ModuleList([nn.Embedding(self.codebook_length, self.latent_dim, device=cst.DEVICE) for _ in range(self.num_quantizers)])
            self.codebooks_freq = torch.zeros(self.num_quantizers, self.codebook_length, device=cst.DEVICE)
            for i in range(self.num_quantizers):
                init.xavier_normal_(self.codebooks[i].weight.data)
                self.codebooks[i].weight.requires_grad = True
 
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
                batch_size=config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE],
                emb_sample_len=self.emb_sample_len,
                num_convs=num_convs,
                dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT],
            )
        

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
        #adding channel dimension
        #x = rearrange(x, 'b l -> b 1 l')
        z_e = self.encoder(x)
        if batch_idx % 10 == 0 and batch_idx != 0 and is_train:
            self._reinitialize(z_e, batch_idx)
        # quantize the latent space
        quantizations, _, comm_loss = self.quantize(z_e, self.num_quantizers, is_train)
        final_z_q = torch.sum(quantizations, dim=0)
        z_q_rearranged = rearrange(final_z_q, '(b l) c -> b l c', b=x.shape[0]).contiguous()
        recon = self.decoder(z_q_rearranged)

        return recon, comm_loss
    

    def quantize(self, x, num_quant, is_train):
        first_z_e = rearrange(x, 'b w c -> (b w) c').contiguous()
        # if it is the first iteration we initialize the codebook with kmeans
        quantizations = []
        list_encodings = []
        self.last_residuals = []
        q_latent_loss = torch.tensor(0, dtype=torch.float32, device=cst.DEVICE)
        rq_latent_loss = torch.tensor(0, dtype=torch.float32, device=cst.DEVICE)
        e_latent_loss = torch.tensor(0, dtype=torch.float32, device=cst.DEVICE)
        z_d = torch.zeros_like(first_z_e)
        z_d.requires_grad = False
        for i in range(num_quant):
            j = i if not self.shared_codebook else 0
            if i == 0:
                z_e = first_z_e
            self.last_residuals.append(rearrange(z_e, '(b w) c -> b w c', b=x.shape[0]).contiguous())
            dist = torch.cdist(z_e, self.codebooks[j].weight)
            _, encoding_indixes = torch.min(dist, dim=1)
            list_encodings.append(encoding_indixes)
            #update the dictionary of indices with the frequency of each index of the codebook
            self.codebooks_freq[j, encoding_indixes] += 1
            # get the corresponding codebook vector for each index
            z_q = self.codebooks[j](encoding_indixes)
            z_d = torch.add(z_d, z_q.detach())
            # Coomitment loss
            q_latent_loss += F.mse_loss(z_e.detach(), z_q)      # we train the codebook
            e_latent_loss += F.mse_loss(z_e, z_q.detach())      # we train the encoder
            rq_latent_loss += F.mse_loss(first_z_e, z_d)
            
            if is_train:
                # straight-through gradient estimation: we attach z_q to the computational graph
                # to subsequently pass the gradient unaltered from the decoder to the encoder
                # whatever the gradients you get for the quantized variable z_q they will just be copy-pasted into z_e
                z_q = z_e + (z_q - z_e).detach()

            # get the residual for the next quantization step
            z_e = z_e - z_q 
            quantizations.append(z_q)
            
        self.codebooks_freq *= 0.93
        commitment_loss = 0.75*q_latent_loss + e_latent_loss + rq_latent_loss
        stacked_quantizations = torch.stack(quantizations)
        stacked_encodings = torch.stack(list_encodings)
        return stacked_quantizations, stacked_encodings, commitment_loss


    def loss(self, input, recon, comm_loss):
        # Reconstruction loss is the mse between the input and the reconstruction
        recon_time_term = F.mse_loss(input[:, :4, :], recon[:, :4, :])
        multi_spectral_recon_loss = torch.tensor(0, dtype=torch.float32, device=cst.DEVICE)
        
        for src in range(len(cst.STEMS)):
            for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
                orig_mel, recon_mel = map(mel_transform, (input[:, src, :], recon[:, src, :]))
                log_orig_mel, log_recon_mel = torch.log(orig_mel), torch.log(recon_mel)

                l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim = -2).mean()
                l2_log_mel_loss = vector_norm(log_orig_mel - log_recon_mel, dim = -2).mean()

                multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss
        
        self.recon_time_terms.append(self.recon_loss_weight*recon_time_term.detach().item())
        self.multi_spectral_recon_losses.append(self.multi_spectral_recon_loss_weight*multi_spectral_recon_loss.detach().item())
        self.commitment_losses.append(comm_loss.detach().item())
        return self.multi_spectral_recon_loss_weight*multi_spectral_recon_loss + recon_time_term + comm_loss

    def _reinitialize(self, z_e, batch_idx):
        num_quantizers = self.num_quantizers if not self.shared_codebook else 1
        # reinitialize all the codebook vector that have less than frequency 1
        # with a random vector from the encoder output
        with torch.no_grad():
            for i_codebook in range(num_quantizers):
                # Find the indexes where the frequency is less than 1
                indexes = self.codebooks_freq[i_codebook] < 1
                num_indexes = indexes.sum().item()
                # Generate random indexes for z_e
                rand = random.randint(0, z_e.shape[0]-1)
                j = np.arange(num_indexes) % z_e.shape[1]
                i = np.repeat(range(rand, z_e.shape[0]), z_e.shape[1])
                if i.shape[0] < num_indexes:
                    concat = np.repeat(range(z_e.shape[0]), z_e.shape[1])
                    concat = concat[:num_indexes-i.shape[0]]
                    i = np.append(i, concat)
                else:
                    i = i[:num_indexes]
                # Assign the random elements from z_e to the corresponding weights in the codebook
                self.codebooks[i_codebook].weight.data[indexes] = self.last_residuals[i_codebook][i, j]
                self.codebooks_freq[i_codebook][indexes] = 0
            # reinitialize the frequences dictionary
            #self.codebooks_freq = torch.zeros(num_quantizers, self.codebook_length, device=cst.DEVICE)


