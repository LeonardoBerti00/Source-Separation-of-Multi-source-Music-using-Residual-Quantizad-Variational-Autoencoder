from models.VQVAE.VQVAE import VQVAE
from models.VAE.VAE import VAE
import constants as cst

def pick_autoencoder(config, autoencoder_name):
    if autoencoder_name == "VQVAE":
        return VQVAE(config).to(device=cst.DEVICE)
    elif autoencoder_name == 'VAE':
        return VAE(config).to(device=cst.DEVICE)
    else:
        raise ValueError("Autoencoder not found")

def pick_model(config, model_name):
    pass
