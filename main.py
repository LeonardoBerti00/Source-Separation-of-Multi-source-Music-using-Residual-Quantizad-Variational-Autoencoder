import lightning as L
from repo.models.VAE_MLP import VAE_MLP
from repo.models.VAE_CNN import VAE_CNN
import constants as cst
from dataset import Datamodule, MultiSourceDataset
from config import Configuration
from lightning.pytorch.callbacks import Callback
import torch

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

def run():
    config = Configuration()
    trainer = L.Trainer(
        accelerator="gpu",
        precision="bf16",
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        profiler="advanced",
        callbacks=PrintCallback(),
        num_sanity_val_steps=0,
    )

    train_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHANNEL_SIZE],
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TRAIN,
        stems=cst.STEMS,
    )
    val_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHANNEL_SIZE],
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_VAL,
        stems=cst.STEMS,
    )
    test_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHANNEL_SIZE],
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TEST,
        stems=cst.STEMS,
    )
    data_module = Datamodule(train_set, val_set, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=0)
    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
    #model = VAE_MLP(config).to(cst.DEVICE)
    model = VAE_CNN(config).to(cst.DEVICE)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    run()