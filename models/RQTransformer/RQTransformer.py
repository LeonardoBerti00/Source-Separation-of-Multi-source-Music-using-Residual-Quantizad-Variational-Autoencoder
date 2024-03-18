from einops import rearrange
import numpy
from config import Configuration
import torch
from torch.nn import functional as F
from torch import nn
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time
from torch_ema import ExponentialMovingAverage
import wandb
from utils.utils_transformer import sinusoidal_positional_embedding, load_autoencoder
from models.RQTransformer.Transformer import TransformerEncoder


class RQTransformer(L.LightningModule):

    def __init__(self, config: Configuration, test_num_steps: int):
        super().__init__()

        self.IS_WANDB = config.IS_WANDB
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.training = config.IS_TRAINING
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.test_reconstructions = numpy.zeros((test_num_steps, len(cst.STEMS)))
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.LitModel, _ = load_autoencoder(config.CHOSEN_AE.name)
        self.AE = self.LitModel.AE
        self.emb_sample_len = int(self.AE.emb_sample_len)
        self.latent_dim = int(self.AE.latent_dim)
        self.codebook_length = int(self.AE.codebook_length)
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
            num_layers=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_TRANSFORMER_LAYERS],
            dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        )
        self.spatial_transformer = TransformerEncoder(
            d_model=self.latent_dim,
            num_heads=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_HEADS],
            num_layers=config.HYPER_PARAMETERS[LearningHyperParameter.NUM_TRANSFORMER_LAYERS],
            dropout=config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        )
        self.fc = torch.nn.Linear(self.latent_dim, self.codebook_length)
        self.loss = nn.NLLLoss(reduction='mean')


    def forward(self, x, is_train):
        #adding channel dimension
        x = rearrange(x, 'b l-> b 1 l')
        z_e = self.AE.encoder(x)
        # z_e shape: (batch_size, emb_sample_len, hidden_channels)
        v = torch.zeros(z_e.shape[0], self.emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        p = torch.zeros(z_e.shape[0], self.emb_sample_len, self.num_quantizers, self.codebook_length, device=cst.DEVICE)
        
        # quantize the input
        stacked_quantizations, encodings, _ = self.AE.quantize(z_e, num_quant=self.num_quantizers, is_train=False)
        encodings = rearrange(encodings, 'n (b l) -> b l n', b=x.shape[0])
        cumsum_quantizations = torch.cumsum(stacked_quantizations, dim=0)
        sum_quantizations = cumsum_quantizations[-1]
        final_z_q = rearrange(sum_quantizations, '(b l) c -> b l c', b=x.shape[0]).contiguous()

        # preparing the input for the spatial transformer
        # first element of the sequence is the start sequence
        # the rest of the elements are the sum of the quantizations
        first_token_spatial = self.pos_emb_t[0] + self.start_seq
        tokens_spatial = self.pos_emb_t[1:] + final_z_q[:, :-1]
        tokens_spatial = torch.cat((first_token_spatial.unsqueeze(0), tokens_spatial), dim=0)
        mask_spatial = self.get_mask(self.emb_sample_len)
        # spatial transformer generates the context vector
        h = self.spatial_transformer(tokens_spatial, mask_spatial)

        # preparing the input for the depth transformer
        first_token_depth = torch.add(self.pos_emb_d[0], h)
        cum_sum_residuals = rearrange(cumsum_quantizations, 'n (b l) c -> b l n c', b=x.shape[0]).contiguous()
        tokens_depth = self.pos_emb_d[1:] + cum_sum_residuals[:, :, :-1]
        tokens_depth = torch.cat((first_token_depth.unsqueeze(0), tokens_depth), dim=0)

        # depth transformer predicts the next code and then we get the log probabilities
        mask_depth = self.get_mask(self.num_quantizers)
        tokens_depth = rearrange(tokens_depth, 'b l n c -> (b l) n c')
        out_depth = self.depth_transformer(v, mask_depth)
        out_depth = rearrange(out_depth, '(b l) n c -> b l n c', b=x.shape[0])
        p = F.log_softmax(self.fc(out_depth), dim=-1)

        return p, encodings


    def get_mask(self, emb_sample_len):
        # upper triangular mask
        mask = torch.ones(emb_sample_len, emb_sample_len, device=cst.DEVICE, requires_grad=False)
        mask = mask.triu(diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def inference(self):
        u = torch.zeros(1, self.emb_sample_len, self.latent_dim, device=cst.DEVICE)
        v = torch.zeros(1, self.emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        p = torch.zeros(1, self.emb_sample_len, self.num_quantizers, device=cst.DEVICE, dtype=torch.long)
        z_q = torch.zeros(1, self.emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        for t in range(0, self.emb_sample_len):
            if t == 0:
                u[:, t] = self.start_seq
            else:
                # preparing the input for the spatial transformer
                sum_quantizations = torch.sum(z_q, dim=2)
                u[:, t] = self.pos_emb_t[t] + sum_quantizations
            mask_spatial = self.get_mask(self.emb_sample_len)
            h_t = self.spatial_transformer(u, mask_spatial)[:, t]
            for d in range(self.num_quantizers):
                if d == 0:
                    v[:, t, d] = self.pos_emb_d[d] + h_t
                else:
                    sum_quantizations = torch.sum(z_q, dim=2)
                    v[:, t, d] = self.pos_emb_d[d] + sum_quantizations
                # depth transformer predicts the next code
                mask_depth = self.get_mask(self.num_quantizers)
                depth_out = self.depth_transformer(v[:, t, :], mask_depth)[:, d]
                prob = F.softmax(self.fc(depth_out), dim=-1)
                p[:, t, d] = torch.argmax(prob, dim=-1)
                encodings = F.one_hot(p[:, t, d], self.codebook_length).float()
                # maybe encodings need unsqueeze
                z_q[:, t, d] = torch.matmul(encodings, self.AE.codebook.weight)

        return self.AE.decoder(torch.sum(z_q, dim=2))


    def training_step(self, input, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(input, dim=-2)
        mixture.requires_grad = True
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        output, real = self.forward(mixture, is_train=True)
        output = rearrange(output, 'b l n c -> (b l n) c')
        real = rearrange(real, 'b l n -> (b l n)')
        batch_loss = self.loss(output, real)
        self.train_losses.append(batch_loss.item())
        self.ema.update()
        return batch_loss

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
        print(f'train loss on epoch {self.current_epoch} is {loss}')

    def validation_step(self, input, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(input, dim=-2)
        mixture.requires_grad = True
        recon, reverse_context = self.forward(mixture, is_train=False)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(input, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.val_losses.append(batch_loss_mean.item())
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(input, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(input, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        return batch_loss_mean

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', self.val_loss)
        if self.IS_WANDB:
            wandb.log({'val_ema_loss': loss_ema}, step=self.current_epoch)
        print(f"\n val loss on epoch {self.current_epoch} is {loss}")
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")

    def test_step(self, input, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(input, dim=-2)
        mixture.requires_grad = True
        recon, reverse_context = self.forward(mixture, is_train=False)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(input, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.test_losses.append(batch_loss_mean.item())
        recon = self._to_original_dim(recon)
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[
            batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.test_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        # Testing: with EMA
        '''
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(input, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(input, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.test_ema_losses.append(ema_loss.item())
            recon = self._to_original_dim(recon)
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.test_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        '''
        return batch_loss_mean

    def on_test_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        if self.IS_WANDB:
            wandb.log({'test_loss': loss})
            wandb.log({'test_ema_loss': loss_ema})
        print(f"\n test loss on epoch {self.current_epoch} is {loss}")
        print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")
        numpy.save(cst.RECON_DIR + "/test_reconstructions.npy", self.test_reconstructions)
        numpy.save(cst.RECON_DIR + "/test_ema_reconstructions.npy", self.test_ema_reconstructions)

    def configure_optimizers(self):
        if self.lr == 0.001:
            end_factor = 0.01
        else:
            end_factor = 0.05
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=end_factor, total_iters=self.epochs)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

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