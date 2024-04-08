import os
from einops import rearrange, repeat
from config import Configuration
import torch
from torch.nn import functional as F
from torch import nn
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
import wandb
from utils.utils import save_audio
from utils.utils_transformer import sinusoidal_positional_embedding, load_autoencoder
from models.RQTransformer.Transformer import TransformerEncoder


class RQTransformer(L.LightningModule):

    def __init__(self, config: Configuration):
        super().__init__()

        self.IS_WANDB = config.IS_WANDB
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.training = config.IS_TRAINING
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.min_loss = 10000000
        self.last_loss = 10000000
        self.LitModel, _ = load_autoencoder(config.CHOSEN_AE.name)
        self.AE = self.LitModel.AE
        self.emb_sample_len = int(self.AE.emb_sample_len)
        self.latent_dim = int(self.AE.latent_dim)
        self.codebook_length = int(self.AE.codebook_length)
        self.chosen_model = config.CHOSEN_MODEL
        self.last_path_ckpt = None
        self.filename_ckpt = config.FILENAME_CKPT
        if config.CHOSEN_AE == cst.Autoencoders.RQVAE:
            self.num_quantizers = int(self.AE.num_quantizers)
        else:
            self.num_quantizers = 1

        self.pos_emb_t = sinusoidal_positional_embedding(self.emb_sample_len, self.latent_dim)
        self.pos_emb_d = sinusoidal_positional_embedding(self.num_quantizers, self.latent_dim)
        
        self.start_seq = torch.nn.Parameter(torch.randn(self.latent_dim))
        self.depth_transformer = TransformerEncoder(
            d_model=self.latent_dim,
            num_heads=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_HEADS],
            num_layers=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DEPTH_TRANSFORMER_LAYERS],
            dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        )
        self.spatial_transformer = TransformerEncoder(
            d_model=self.latent_dim,
            num_heads=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_HEADS],
            num_layers=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_SPATIAL_TRANSFORMER_LAYERS],
            dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        )
        self.fc = torch.nn.Linear(self.latent_dim, self.codebook_length)
        self.loss = nn.NLLLoss(reduction='mean')
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)

    def forward(self, x, is_train):
        #adding channel dimension
        x = rearrange(x, 'b t -> b 1 t')
        z_e, _, _, _, _, _ = self.AE.encoder(x)
        # z_e shape: (batch_size, emb_sample_len, hidden_channels)
        p = torch.zeros(z_e.shape[0], self.emb_sample_len, self.num_quantizers, self.codebook_length, device=cst.DEVICE)
        
        # quantize the input
        stacked_quantizations, encodings, _ = self.AE.quantize(z_e, num_quant=self.num_quantizers, is_train=False)
        encodings = rearrange(encodings, 'n (b t) -> b t n', b=x.shape[0])
        cumsum_quantizations = torch.cumsum(stacked_quantizations, dim=0)
        sum_quantizations = cumsum_quantizations[-1]
        sum_quantizations = rearrange(sum_quantizations, '(b t) c -> b t c', b=x.shape[0]).contiguous()

        # preparing the input for the spatial transformer
        # first element of the sequence is the start sequence
        # the rest of the elements are the sum of the quantizations
        start_seq_spatial = repeat(self.start_seq, 'c -> b 1 c', b=x.shape[0])
        input_spatial = torch.cat((start_seq_spatial, sum_quantizations[:, :-1]), dim=1)
        # adding positional spatial encoding
        input_spatial = torch.add(self.pos_emb_t, input_spatial)
        mask_spatial = self.get_mask(self.emb_sample_len)

        # spatial transformer generates the context vector
        h = self.spatial_transformer(input_spatial, mask_spatial)

        # preparing the input for the depth transformer
        # first element of the sequence is the output of the spatial transformer
        # the rest of the elements are the cum sum of the quantizations
        h = rearrange(h, 'b t c -> b t 1 c')
        cumsum_quantizations = rearrange(cumsum_quantizations, 'd (b t) c -> b t d c', b=x.shape[0]).contiguous()
        input_depth = torch.cat((h, cumsum_quantizations[:, :, :-1]), dim=2)
        input_depth = rearrange(input_depth, 'b t d c -> (b t) d c')
        # adding positional depth encoding
        input_depth = torch.add(self.pos_emb_d, input_depth)

        # depth transformer predicts the next code and then we get the log probabilities
        mask_depth = self.get_mask(self.num_quantizers)
        out_depth = self.depth_transformer(input_depth, mask_depth)
        out_depth = rearrange(out_depth, '(b t) d c -> b t d c', b=x.shape[0])
        log_logits = F.log_softmax(self.fc(out_depth), dim=-1)

        return log_logits, encodings


    def get_mask(self, emb_sample_len):
        # upper triangular mask
        mask = torch.ones(emb_sample_len, emb_sample_len, device=cst.DEVICE, requires_grad=False)
        mask = mask.triu(diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def sampling(self, batch_size, duration, T=1.0):
        emb_sample_len = duration * self.emb_sample_len // cst.DURATION
        u = torch.zeros(batch_size, emb_sample_len, self.latent_dim, device=cst.DEVICE)
        v = torch.zeros(batch_size, emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        p = torch.zeros(batch_size, emb_sample_len, self.num_quantizers, device=cst.DEVICE, dtype=torch.long)
        z_q = torch.zeros(batch_size, emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        for t in range(emb_sample_len):
            if t == 0:
                u[:, t] = self.start_seq
            else:
                # preparing the input for the spatial transformer
                sum_quantizations = torch.sum(z_q[:, t-1], dim=1)
                u[:, t] = self.pos_emb_t[t] + sum_quantizations
            mask_spatial = self.get_mask(emb_sample_len)
            h_t = self.spatial_transformer(u, mask_spatial)[:, t]
            for d in range(self.num_quantizers):
                if d == 0:
                    v[:, t, d] = self.pos_emb_d[d] + h_t
                else:
                    sum_quantizations_d = torch.sum(z_q[:, t], dim=1)
                    v[:, t, d] = self.pos_emb_d[d] + sum_quantizations_d
                # depth transformer predicts the next code
                mask_depth = self.get_mask(self.num_quantizers)
                input_depth = rearrange(v, 'b t d c -> (b t) d c')
                depth_out = self.depth_transformer(input_depth, mask_depth)
                depth_out = rearrange(depth_out, '(b t) d c -> b t d c', b=batch_size)
                depth_out_td = depth_out[:, t, d]
                logits_td = F.softmax(self.fc(depth_out_td)//T, dim=-1)
                sample = self.top_p_sampling(logits_td)
                p[:, t, d] = sample
                encodings = F.one_hot(p[:, t, d], self.codebook_length).float()
                if self.AE.shared_codebook:
                    z_q[:, t, d] = torch.matmul(encodings, self.AE.codebooks[0].weight)
                else:
                    z_q[:, t, d] = torch.matmul(encodings, self.AE.codebooks[d].weight)

        return self.AE.decoder(torch.sum(z_q, dim=2), 0, 0, 0, 0, 0)

    def top_p_sampling(self, logits, p=0.9):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.shape[0]):
            indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = 0.0
        return torch.multinomial(logits, 1, replacement=True)[:, 0]

    def training_step(self, input, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(input, dim=-2)
        mixture.requires_grad = True
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        log_logits, real = self.forward(mixture, is_train=True)
        log_logits = rearrange(log_logits, 'b t d c -> (b t d) c')
        real = rearrange(real, 'b t d -> (b t d)')
        batch_loss = self.loss(log_logits, real)
        self.train_losses.append(batch_loss.item())
        self.ema.update()
        return batch_loss

    def on_train_epoch_end(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
        print(f'train loss on epoch {self.current_epoch} is {loss}')
        self.train_losses = []
        self.val_losses = []
        self.val_ema_losses = []

    def validation_step(self, input, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(input, dim=-2)
        mixture.requires_grad = True
        with self.ema.average_parameters():
            output, real = self.forward(mixture, is_train=False)
            output = rearrange(output, 'b t d c -> (b t d) c')
            real = rearrange(real, 'b t d -> (b t d)')
            ema_loss = self.loss(output, real)
            self.val_ema_losses.append(ema_loss.item())
        return ema_loss

    def on_validation_epoch_end(self) -> None:
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_ema_loss', loss_ema)
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")
        # model checkpointing
        if loss_ema < self.min_loss:
            self.min_loss = loss_ema
            self._model_checkpointing(loss_ema)
        # if the improvement is less than 0.005, we halve the learning rate 
        if loss_ema - self.last_loss > 0.005:
            self.optimizer.param_groups[0]["lr"] /= 2
        self.last_loss = loss_ema

    def test_step(self, input, batch_idx):
        mixture = torch.sum(input, dim=-2)
        with self.ema.average_parameters():
            output, real = self.forward(mixture, is_train=False)
            output = rearrange(output, 'b t d c -> (b t d) c')
            real = rearrange(real, 'b t d -> (b t d)')
            ema_loss = self.loss(output, real)
            self.test_ema_losses.append(ema_loss.item())
        return ema_loss

    def on_test_end(self) -> None:
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        if self.IS_WANDB:
            wandb.log({'test_ema_loss': loss_ema})
        #print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")
        
    def generating(self):
        print("starting generation of 16 samples with duration 4s and 8s")
        generated = self.sampling(batch_size=16, duration=4)
        for i in range(16):
            save_audio(generated[i][0], 'recon_bass'+str(i))
            save_audio(generated[i][1], 'recon_drums'+str(i))
            save_audio(generated[i][2], 'recon_guitar'+str(i))
            save_audio(generated[i][3], 'recon_piano'+str(i))
            save_audio((generated[i][0]+generated[i][1]+generated[i][2]+generated[i][3])//4, 'recon_mix'+str(i))
        generated8s = self.sampling(batch_size=16, duration=8)
        for i in range(16):
            save_audio(generated8s[i][0], 'recon8s_bass'+str(i))
            save_audio(generated8s[i][1], 'recon8s_drums'+str(i))
            save_audio(generated8s[i][2], 'recon8s_guitar'+str(i))
            save_audio(generated8s[i][3], 'recon8s_piano'+str(i))
            save_audio((generated8s[i][0]+generated8s[i][1]+generated8s[i][2]+generated8s[i][3])//4, 'recon8s_mix'+str(i))

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer

    def inference_time(self, x):
        t0 = time.time()
        _ = self.forward(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")
        wandb.define_metric("test_loss", summary="min")
        wandb.define_metric("test_ema_loss", summary="min")

    def _model_checkpointing(self, loss):
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        with self.ema.average_parameters():
            filename_ckpt = ("val_loss=" + str(round(loss, 4)) +
                                "_epoch=" + str(self.current_epoch) +
                                "_" + self.filename_ckpt
                                )
            path_ckpt = cst.DIR_SAVED_MODEL + "/" + self.chosen_model.name + "/" + filename_ckpt
            self.trainer.save_checkpoint(path_ckpt)
            self.last_path_ckpt = path_ckpt