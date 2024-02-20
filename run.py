import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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


HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Autoencoders.VQVAE: HP_SEARCH_TYPES(HP_VQVAE, HP_VQVAE_FIXED),
    cst.Autoencoders.RQVAE: HP_SEARCH_TYPES(HP_RQVAE, HP_RQVAE_FIXED)
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
        callbacks=[
            EarlyStopping(monitor="val_snr", mode="max", patience=4, verbose=False)
        ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler="simple"
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
            # log simulation details in WANDB console
            wandb_instance.log({"model": config.CHOSEN_MODEL.name}, commit=False)
            wandb_instance.log({"sr": cst.SAMPLE_RATE}, commit=False)
            wandb_instance.log({"conv_setup": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]}, commit=False)
            wandb_instance.log({"batch size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]}, commit=False)
            wandb_instance.log({"dropout": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DROPOUT]}, commit=False)
            wandb_instance.log({"lstm layers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LSTM_LAYERS]}, commit=False)
            wandb_instance.log({"latent dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.HIDDEN_CHANNELS][config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]][-1]}, commit=False)
            wandb_instance.log({"num convs": config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_CONVS]}, commit=False)
            wandb_instance.log({"sample rate": cst.SAMPLE_RATE}, commit=False)

            if config.IS_SWEEP:
                model_params = wandb.config

            wandb_instance_name = ""

            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    config.HYPER_PARAMETERS[param] = model_params[param.value]
                    wandb_instance_name += str(param.value[:3]) + "_" + str(model_params[param.value]) + "_"

            sr = cst.SAMPLE_RATE
            conv_setup = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]
            config.FILENAME_CKPT = "convsetup_" + str(conv_setup) + "_sr_" + str(sr) + "_model_" + str(config.CHOSEN_MODEL.name) + "_ckpt.ckpt"

            trainer = L.Trainer(
                accelerator=accelerator,
                precision=cst.PRECISION,
                max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
                callbacks=[EarlyStopping(monitor="val_snr", mode="max", patience=4, verbose=True, min_delta=0.001)],
                num_sanity_val_steps=0,
                logger=wandb_logger,
                detect_anomaly=False,
            )
            train(config, trainer)

    return wandb_sweep_callback


def sweep_init(config):
    wandb.login(key="d29d51017f4231b5149d36ad242526b374c9c60a")
    sweep_config = {
        'method': 'grid',
        'metric': {
            'goal': 'minimize',
            'name': 'val_ema_loss'
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
