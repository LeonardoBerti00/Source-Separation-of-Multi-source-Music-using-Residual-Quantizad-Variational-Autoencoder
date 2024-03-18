import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
import constants as cst
from dataset import DataModule, MultiSourceDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from collections import namedtuple
from models.LitAutoencoder import LitAutoencoder

from models.RQTransformer.RQTransformer import RQTransformer
from models.VQVAE.VQVAE_hp import HP_VQVAE, HP_VQVAE_FIXED
from models.RQVAE.RQVAE_hp import HP_RQVAE, HP_RQVAE_FIXED
from models.VAE.VAE_hp import HP_VAE, HP_VAE_FIXED
from utils.utils import compute_centroids


HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Autoencoders.VQVAE: HP_SEARCH_TYPES(HP_VQVAE, HP_VQVAE_FIXED),
    cst.Autoencoders.RQVAE: HP_SEARCH_TYPES(HP_RQVAE, HP_RQVAE_FIXED),
    cst.Autoencoders.VAE: HP_SEARCH_TYPES(HP_VAE, HP_VAE_FIXED),
}


def train(config, trainer):
    print_setup(config)

    train_set = MultiSourceDataset(
        sr=cst.SAMPLE_RATE,
        channels=cst.CHANNEL_SIZE,
        min_duration=cst.MIN_DURATION,
        max_duration=cst.MAX_DURATION,
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=cst.SAMPLE_LENGTH,
        audio_files_dir=cst.AUDIO_FILES_DIR_TRAIN,
        stems=cst.STEMS,
        z_score=config.HYPER_PARAMETERS[cst.LearningHyperParameter.Z_SCORE]
    )

    val_set = MultiSourceDataset(
        sr=cst.SAMPLE_RATE,
        channels=cst.CHANNEL_SIZE,
        min_duration=cst.MIN_DURATION,
        max_duration=cst.MAX_DURATION,
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=cst.SAMPLE_LENGTH,
        audio_files_dir=cst.AUDIO_FILES_DIR_VAL,
        stems=cst.STEMS,
        z_score=config.HYPER_PARAMETERS[cst.LearningHyperParameter.Z_SCORE]
    )

    test_set = MultiSourceDataset(
        sr=cst.SAMPLE_RATE,
        channels=cst.CHANNEL_SIZE,
        min_duration=cst.MIN_DURATION,
        max_duration=cst.MAX_DURATION,
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=cst.SAMPLE_LENGTH,
        audio_files_dir=cst.AUDIO_FILES_DIR_TEST,
        stems=cst.STEMS,
        z_score=config.HYPER_PARAMETERS[cst.LearningHyperParameter.Z_SCORE]
    )

    data_module = DataModule(
        train_dataset=train_set, 
        val_dataset=val_set, 
        test_dataset=test_set, 
        batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], 
        num_workers=8
    )

    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    if config.IS_TRAINING_AE:
        model = LitAutoencoder(config).to(cst.DEVICE, torch.float32)
        if config.CHOSEN_MODEL == cst.Autoencoders.VQVAE or config.CHOSEN_MODEL == cst.Autoencoders.RQVAE:
            code_len = model.AE.codebook_length
            latent_dim = model.AE.latent_dim
            if config.HYPER_PARAMETERS[cst.LearningHyperParameter.INIT_KMEANS]:
                #check if there is a ifle called centroids in data
                if not os.path.isfile(f"data/centroids{code_len}_{latent_dim}.npy"):
                    # compute centroids, this will take some minutes
                    centroids = compute_centroids(model, train_set)
                    # save centroids in the directory data
                    np.save(f"data/centroids{code_len}_{latent_dim}.npy", centroids)
                else:
                    # centroids len is 16184
                    # load centroids in the directory data
                    centroids = np.load(f"data/centroids{code_len}_{latent_dim}.npy")
                if centroids.shape[0] < code_len:
                    model.AE.codebooks[0].weight.data[:centroids.shape[0]] = torch.tensor(centroids, device=cst.DEVICE, dtype=torch.float32).contiguous()
                else:
                    model.AE.codebooks[0].weight.data = torch.tensor(centroids[:code_len], device=cst.DEVICE, dtype=torch.float32).contiguous()
    else:
        model = RQTransformer(config=config, test_num_steps=test_set.__len__()).to(cst.DEVICE, torch.float32)

    print("\nstarting training\n")
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

def test(config, trainer, model):
    print_setup(config)

    test_set = MultiSourceDataset(
        sr=cst.SAMPLE_RATE,
        channels=cst.CHANNEL_SIZE,
        min_duration=cst.MIN_DURATION,
        max_duration=cst.MAX_DURATION,
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=cst.SAMPLE_LENGTH,
        audio_files_dir=cst.AUDIO_FILES_DIR_TEST,
        stems=cst.STEMS,
        z_score=config.HYPER_PARAMETERS[cst.LearningHyperParameter.Z_SCORE]
    )

    data_module = DataModule(None, None, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=8)

    test_dataloader = data_module.test_dataloader()

    model.to(cst.DEVICE, torch.float32)

    trainer.test(model, dataloaders=test_dataloader)


def run(config, accelerator, model=None):
    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        callbacks=[EarlyStopping(monitor="val_ema_sdr", mode="max", patience=3, verbose=False)],
        num_sanity_val_steps=0,
        detect_anomaly=False,
    )
    if (config.IS_TESTING):
        test(config, trainer, model)
    else:
        train(config, trainer)


def run_wandb(config, accelerator):
    def wandb_sweep_callback():
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        if not config.IS_SWEEP:
            run_name = ""
            model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    run_name += str(param.value[:3]) + "_" + str(model_params[param.value]) + "_"

        with wandb.init(project=cst.PROJECT_NAME, name=run_name) as wandb_instance:
            if config.IS_SWEEP:
                model_params = wandb.config

            wandb_instance_name = ""
            config.FILENAME_CKPT = ""

            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    config.HYPER_PARAMETERS[param] = model_params[param.value]
                    wandb_instance_name += str(param.value[:3]) + "_" + str(model_params[param.value]) + "_"
                    config.FILENAME_CKPT += str(param.value[:3]) + "_" + str(model_params[param.value]) + "_"

            sr = cst.SAMPLE_RATE
            conv_setup = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]
            res_type = config.HYPER_PARAMETERS[cst.LearningHyperParameter.RES_TYPE]
            config.FILENAME_CKPT += "convsetup_" + str(conv_setup) + "_sr_" + str(sr) + "_duration_" + str(cst.DURATION) + "_restype_" + str(res_type) + ".ckpt"

            trainer = L.Trainer(
                accelerator=accelerator,
                precision=cst.PRECISION,
                max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
                callbacks=[EarlyStopping(monitor="val_ema_sdr", mode="max", patience=3, verbose=True, min_delta=0.1)],
                num_sanity_val_steps=0,
                logger=wandb_logger,
                detect_anomaly=False,
            )
            # log simulation details in WANDB console
            wandb_instance.log({"model": config.CHOSEN_MODEL.name}, commit=False)
            wandb_instance.log({"sr": cst.SAMPLE_RATE}, commit=False)
            wandb_instance.log({"conv_setup": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]}, commit=False)
            wandb_instance.log({"batch size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]}, commit=False)
            wandb_instance.log({"dropout": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DROPOUT]}, commit=False)
            wandb_instance.log({"lstm layers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LSTM_LAYERS]}, commit=False)
            wandb_instance.log({"latent dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.HIDDEN_CHANNELS][config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]][-1]}, commit=False)
            wandb_instance.log({"num convs": config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_CONVS]}, commit=False)
            wandb_instance.log({"duration": cst.DURATION}, commit=False)
            wandb_instance.log({"commitment loss weight": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SDR_LOSS_WEIGHT]}, commit=False)
            wandb_instance.log({"recon loss weight": config.HYPER_PARAMETERS[cst.LearningHyperParameter.RECON_LOSS_WEIGHT]}, commit=False)
            wandb_instance.log({"multi spectral recon loss weight": config.HYPER_PARAMETERS[cst.LearningHyperParameter.MULTI_SPECTRAL_RECON_LOSS_WEIGHT]}, commit=False)
            wandb_instance.log({"learning rate": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE]}, commit=False)
            wandb_instance.log({"code len": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CODEBOOK_LENGTH]}, commit=False)
            wandb_instance.log({"num quantizers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_QUANTIZERS]}, commit=False)
            wandb_instance.log({"optimizer": config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER]}, commit=False)
            wandb_instance.log({"shared codek": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SHARED_CODEBOOK]}, commit=False)
            train(config, trainer)

    return wandb_sweep_callback


def sweep_init(config):
    wandb.login(key="d29d51017f4231b5149d36ad242526b374c9c60a")
    sweep_config = {
        'method': 'grid',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
            'eta': 1.5
        },
        'run_cap': 100,
        'parameters': {**HP_DICT_MODEL[config.CHOSEN_MODEL].sweep}
    }
    return sweep_config

def print_setup(config):
    print(f"Chosen model: {config.CHOSEN_MODEL.name}")
    print(f"Duration: {cst.DURATION}")
    print(f"Sample rate: {cst.SAMPLE_RATE}")
    print(f"Convolution setup: {config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]}")
