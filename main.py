import wandb
from models.VQVAE.VQVAE import VQVAE
from run import run_wandb, run, sweep_init
import torch
import constants as cst
from config import Configuration
from utils.utils_models import load_transformer
from utils.utils_transformer import load_autoencoder


def set_torch():
    torch.manual_seed(cst.SEED)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    set_torch()
    
    config = Configuration()
    if cst.DEVICE == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "gpu"

    if config.IS_WANDB:
        #wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model="all", save_dir=cst.DIR_SAVED_MODEL)
        if config.IS_SWEEP:
            sweep_config = sweep_init(config)
            sweep_config.update({"name": 
                                    f"model_{config.CHOSEN_MODEL.name}"
                                    f"_sl_{cst.SAMPLE_LENGTH}"
                                    f"_conv_setup_{config.HYPER_PARAMETERS[cst.LearningHyperParameter.CONV_SETUP]}"
                                })
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME)
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    elif config.IS_TESTING:
        if config.CHOSEN_MODEL == cst.Models.RQTransformer:
            model, config = load_transformer(config.CHOSEN_MODEL.name)
        else:
            model, config = load_autoencoder(config.CHOSEN_MODEL.name)
        config.IS_WANDB = False
        config.IS_TRAINING = False
        config.IS_TESTING = True     
        run(config, accelerator, model)

    # training without using wandb
    elif config.IS_TRAINING:
        run(config, accelerator)