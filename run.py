import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger
import constants as cst
from dataset import DataModule, MultiSourceDataset
from config import Configuration
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from collections import namedtuple

from models.LitModel import LitModel
from models.VQVAE.VQVAE_hp import HP_VQVAE, HP_VQVAE_FIXED
from utils.utils_models import pick_autoencoder

HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Autoencoders.VQVAE: HP_SEARCH_TYPES(HP_VQVAE, HP_VQVAE_FIXED)
}

def train(config, trainer):
    print_setup(config)

    train_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=cst.CHANNEL_SIZE,
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TRAIN,
        stems=cst.STEMS,
    )

    val_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=cst.CHANNEL_SIZE,
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_VAL,
        stems=cst.STEMS,
    )

    test_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=cst.CHANNEL_SIZE,
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TEST,
        stems=cst.STEMS,
    )

    data_module = DataModule(train_set, val_set, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=16)
    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    if config.IS_TRAINING_AE:
        model = pick_autoencoder(config, config.CHOSEN_AE.name).to(cst.DEVICE, torch.float32)
    else:
        model = LitModel(
            config=config,
            val_num_steps=val_set.__len__(),
            test_num_steps=test_set.__len__(),
            trainer=trainer
        ).to(cst.DEVICE, torch.float32)

    print("\nstarting training\n")
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

def test(config, trainer, model):
    print_setup(config)

    test_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=cst.CHANNEL_SIZE,
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TEST,
        stems=cst.STEMS,
    )

    data_module = DataModule(None, None, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=16)

    test_dataloader = data_module.test_dataloader()

    model.to(cst.DEVICE, torch.float32)

    trainer.test(model, dataloaders=test_dataloader)


def run(config, accelerator, model=None):
    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        profiler="advanced",
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)
        ],
        num_sanity_val_steps=0,
    )
    if (config.IS_TESTING):
        test(config, trainer, model)
    else:
        train(config, trainer)


def run_wandb(config, accelerator, wandb_logger):
    def wandb_sweep_callback():
        run_name = None
        if not config.IS_SWEEP:
            run_name = ""
            model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    run_name += str(param.value) + "_" + str(model_params[param.value]) + "_"

        with wandb.init(project=cst.PROJECT_NAME, name=run_name) as wandb_instance:
            # log simulation details in WANDB console
            wandb_instance.log({"model": config.CHOSEN_MODEL.name}, commit=False)

            config.WANDB_INSTANCE = wandb_instance
            model_params = wandb.config

            if config.IS_SWEEP:
                model_params = wandb.config

            wandb_instance_name = ""

            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    config.HYPER_PARAMETERS[param] = model_params[param.value]
                    wandb_instance_name += str(param.value) + "_" + str(model_params[param.value]) + "_"

            checkpoint_callback = ModelCheckpoint(
                dirpath=cst.DIR_SAVED_MODEL,
                monitor="val_loss",
                mode="min",
                save_last=True,
                save_top_k=1,
                every_n_epochs=1,
                filename=str(config.CHOSEN_MODEL.name)+"/{val_loss:.2f}_{epoch}_"+wandb_instance_name
            )
            checkpoint_callback.CHECKPOINT_NAME_LAST = str(config.CHOSEN_MODEL.name)+"/{val_loss:.2f}_{epoch}_"+wandb_instance_name+"_last"

            trainer = L.Trainer(
                accelerator=accelerator,
                precision=cst.PRECISION,
                max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
                profiler="advanced",
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True),
                    checkpoint_callback,
                ],
                num_sanity_val_steps=0,
                logger=wandb_logger,
            )
            train(config, trainer)
    return wandb_sweep_callback


def sweep_init(config):
    #wandb.login("d29d51017f4231b5149d36ad242526b374c9c60a")
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
            'eta': 1.5
        },
        'run_cap': 10,
        'parameters': {**HP_DICT_MODEL[config.CHOSEN_MODEL].sweep}
    }
    return sweep_config

def print_setup(config):
    print(f"Chosen model: {config.CHOSEN_MODEL.name}")
