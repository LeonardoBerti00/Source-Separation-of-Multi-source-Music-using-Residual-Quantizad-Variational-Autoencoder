from einops import rearrange
import numpy
from config import Configuration
import torch
from torch.nn import functional as F
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

    def forward(self, x, is_train):
        #adding channel dimension
        x = rearrange(x, 'b l-> b 1 l')
        z_e = self.AE.encoder(x)
        # z_e shape: (batch_size, emb_sample_len, hidden_channels)
        u = torch.zeros(z_e.shape[0], self.emb_sample_len, self.latent_dim, device=cst.DEVICE)
        v = torch.zeros(z_e.shape[0], self.emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        p = torch.zeros(z_e.shape[0], self.emb_sample_len, self.num_quantizers, device=cst.DEVICE)
        for t in range(0, self.emb_sample_len):
            if t == 0:
                u[:, t] = self.start_seq
            else:
                # preparing the input for the spatial transformer
                _, stacked_quantizations = self.AE.quantize(z_e[:, t-1], num_quant=self.num_quantizers, is_train=False)
                sum_quantizations = torch.sum(stacked_quantizations, dim=0)
                u[:, t] = self.pos_emb_t[t] + sum_quantizations
            mask = self.get_mask(t, self.emb_sample_len)
            h_t = self.spatial_transformer(u, mask)[:, t]
            for d in range(self.num_quantizers):
                if d == 0:
                    v[:, t, d] = self.pos_emb_d[d] + h_t
                else:
                    _, stacked_quantizations = self.AE.quantize(z_e[:, t], num_quant=d, is_train=False)
                    sum_quantizations = torch.sum(stacked_quantizations, dim=0)
                    v[:, t, d] = self.pos_emb_d[d] + sum_quantizations
                # depth transformer predicts the next code
                p[:, t, d] = self.depth_transformer(v, mask)
        return p

    def get_mask(self, t, emb_sample_len):
        mask = torch.zeros(emb_sample_len, emb_sample_len, device=cst.DEVICE, requires_grad=False)
        row_mask = torch.arange(emb_sample_len).unsqueeze(1) > t
        col_mask = torch.arange(emb_sample_len).unsqueeze(0) > t

        # Combine the masks using logical OR
        indexes = row_mask | col_mask

        # Apply the mask to the matrix
        mask[indexes] = float('-inf')
        return mask

    def inference(self):
        u = torch.zeros(1, self.emb_sample_len, self.latent_dim, device=cst.DEVICE)
        v = torch.zeros(1, self.emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        p = torch.zeros(1, self.emb_sample_len, self.num_quantizers, device=cst.DEVICE)
        z_q = torch.zeros(1, self.emb_sample_len, self.num_quantizers, self.latent_dim, device=cst.DEVICE)
        for t in range(0, self.emb_sample_len):
            if t == 0:
                u[:, t] = self.start_seq
            else:
                # preparing the input for the spatial transformer
                sum_quantizations = torch.sum(z_q, dim=2)
                u[:, t] = self.pos_emb_t[t] + sum_quantizations
            h_t = self.spatial_transformer(u, t, is_train=False)
            for d in range(self.num_quantizers):
                if d == 0:
                    v[:, t, d] = self.pos_emb_d[d] + h_t
                else:
                    sum_quantizations = torch.sum(z_q, dim=2)
                    v[:, t, d] = self.pos_emb_d[d] + sum_quantizations
                # depth transformer predicts the next code
                p[:, t, d] = self.depth_transformer(v, t, d, is_train=False)
                encodings = F.one_hot(p[:, t, d], self.AE.codebook_length).float()
                # maybe encodings need unsqueeze
                z_q[:, t, d] = torch.matmul(encodings, self.AE.codebook.weight)


    def loss(self, real, recon, **kwargs):
        return self.NN.loss(real, recon, **kwargs)

    def training_step(self, input, batch_idx):
        with torch.no_grad():
            # from source to mixture
            mixture = torch.sum(input, dim=-2)
        mixture.requires_grad = True
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        output, reverse_context = self.forward(mixture, is_train=True)
        self.inference()
        reverse_context.update({'is_train': True})
        batch_losses = self.loss(input, output, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        return batch_loss_mean

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