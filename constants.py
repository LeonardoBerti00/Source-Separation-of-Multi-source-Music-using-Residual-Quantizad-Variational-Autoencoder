from enum import Enum

import torch


class LearningHyperParameter(str, Enum):
    OPTIMIZER = "optimizer_name"
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"
    EPS = "eps"
    MOMENTUM = "momentum"
    EPOCHS = "epochs"
    BATCH_SIZE = "batch_size"
    DROPOUT = "dropout"
    SAMPLE_RATE = "sample_rate"
    DURATION = "duration"
    SAMPLE_LENGTH = "sample_length"
    CHANNEL_SIZE = "channel_size"
    AUG_SHIFT = "aug_shift"
    MIN_DURATION = "min_duration"
    MAX_DURATION = "max_duration"
    LATENT_DIM = "latent_dim"
    HIDDEN_MLP = "hidden_mlp"
    HIDDEN_CHANNELS = "hidden_channels"
    KERNEL_SIZES = "kernel_sizes"
    STRIDES = "strides"
    DILATIONS = "dilations"
    PADDINGS = "paddings"
    CODEBOOK_LENGTH = "codebook_length"
    LSTM_LAYERS = "lstm_layers"
    COMMITMENT_COST = "commitment_cost"


class Optimizers(Enum):
    ADAM = "Adam"
    LION = "LION"


class Autoencoders(str, Enum):
    VAE = "VAE"
    VQVAE = "VQVAE"
    RQVAE = "RQVAE"


class Transformers(str, Enum):
    TRANSFORMER = "Transformer"


SEED = 0

PRECISION = 32

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUDIO_FILES_DIR_TRAIN = 'data/train'
AUDIO_FILES_DIR_VAL = 'data/validation'
AUDIO_FILES_DIR_TEST = 'data/test'
STEMS = ["bass", "drums", "guitar", "piano"]
WANDB_DIR = "data/wandb"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"

PROJECT_NAME = "MMLM"
