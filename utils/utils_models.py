from pathlib import Path
import torch

import constants as cst
from models.RQTransformer.RQTransformer import RQTransformer


def load_transformer(model_name):
    dir = Path(cst.DIR_SAVED_MODEL + "/" + model_name)
    best_val_loss = 100000

    for file in dir.iterdir():
        val_loss = float(file.name.split("=")[1].split("_")[0])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_reference = file

    # load checkpoint
    checkpoint = torch.load(checkpoint_reference, map_location=cst.DEVICE) 
    config = checkpoint["hyper_parameters"]["config"]  
    model = RQTransformer.load_from_checkpoint(checkpoint_reference, map_location=cst.DEVICE)
    model.IS_WANDB = False
    model.is_training = False
    # we freeze the model
    for param in model.parameters():
        param.requires_grad = False
    return model, config



        

