from pathlib import Path
import torch
import constants as cst
from models.LitAutoencoder import LitAutoencoder


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings.to(cst.DEVICE, non_blocking=True)


def load_autoencoder(model_name):
    dir = Path(cst.DIR_SAVED_MODEL + "/" + model_name)
    best_val_loss = -100000

    for file in dir.iterdir():
        val_loss = float(file.name.split("=")[1].split("_")[0])
        if val_loss == 2.4309:
        #if val_loss > best_val_loss:
            best_val_loss = val_loss
            checkpoint_reference = file
    print("loading checkpoint: ", checkpoint_reference)
    # load checkpoint
    checkpoint = torch.load(checkpoint_reference, map_location=cst.DEVICE) 
    config = checkpoint["hyper_parameters"]["config"]  
 
    model = LitAutoencoder.load_from_checkpoint(checkpoint_reference, map_location=cst.DEVICE)
    model.IS_DEBUG = True
    model.IS_WANDB = False
    model.is_training = False
    model.AE.init_kmeans = False
    config.IS_TESTING = True
    # we freeze the model
    for param in model.parameters():
        param.requires_grad = False
    return model, config