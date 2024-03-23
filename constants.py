from enum import Enum
import torch


class LearningHyperParameter(str, Enum):
    OPTIMIZER = "optimizer_name"
    LEARNING_RATE = "lr"
    EPOCHS = "epochs"
    BATCH_SIZE = "batch_size"
    DROPOUT = "dropout"
    SAMPLE_RATE = "sample_rate"
    SAMPLE_LENGTH = "sample_length"
    AUG_SHIFT = "aug_shift"
    LATENT_DIM = "latent_dim"
    HIDDEN_CHANNELS = "hidden_channels"
    KERNEL_SIZES = "kernel_sizes"
    STRIDES = "strides"
    DILATIONS = "dilations"
    PADDINGS = "paddings"
    CODEBOOK_LENGTH = "codebook_length"
    LSTM_LAYERS = "lstm_layers"
    BETA = "beta"
    NUM_CONVS = "num_convs"
    IS_RESIDUAL = "is_residual"
    INIT_KMEANS = "init_kmeans"
    CONV_SETUP = "conv_setup"
    MULTI_SPECTRAL_RECON_LOSS_WEIGHT = "multi_spectral_recon_loss_weight"
    SDR_LOSS_WEIGHT = "SDR_LOSS_WEIGHT"
    RES_TYPE = "res_type"
    NUM_QUANTIZERS = "num_quantizers"
    NUM_HEADS = "num_heads"
    NUM_TRANSFORMER_LAYERS = "num_transformer_layers"
    Z_SCORE = "z_score"
    RECON_LOSS_WEIGHT = "recon_loss_weight"
    SHARED_CODEBOOK = "shared_codebook"
    NUM_TRANS_AE_LAYERS = "num_trans_ae_layers"
    P = "p"
    NUM_STEPS = "num_steps"


class Optimizers(Enum):
    ADAM = "Adam"
    LION = "LION"


class Autoencoders(str, Enum):
    VAE = "VAE"
    VQVAE = "VQVAE"
    RQVAE = "RQVAE"


class Transformers(str, Enum):
    RQTRANSFORMER = "RQTransformer"


SEED = 0

PRECISION = 32
CHANNEL_SIZE = 1
DURATION = 4
MIN_DURATION = 12
MAX_DURATION = 640
SAMPLE_RATE = 22050
SAMPLE_LENGTH = SAMPLE_RATE * DURATION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUDIO_FILES_DIR_TRAIN = 'data/Slakh/slakh2100/train'
AUDIO_FILES_DIR_VAL = 'data/Slakh/slakh2100/validation'
AUDIO_FILES_DIR_TEST = 'data/Slakh/slakh2100/test'
#AUDIO_FILES_DIR_TRAIN = 'data/babySlakh/train'
#AUDIO_FILES_DIR_VAL = 'data/babySlakh/validation'
#AUDIO_FILES_DIR_TEST = 'data/babySlakh/test'
STEMS = ["bass", "drums", "guitar", "piano"]
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"

PROJECT_NAME = "MMLM"

MEAN = 122.759
STD = 2359.9309

MULTI_SPECTRAL_N_FFTS = 512
MULTI_SPECTRAL_N_MELS = 64
MULTI_SPECTRAL_WINDOW_POWERS = tuple(range(6, 12))