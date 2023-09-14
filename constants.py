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
    HIDDEN1_MLP = "hidden1_mlp"
    HIDDEN2_MLP = "hidden2_mlp"
    HIDDEN3_MLP = "hidden3_mlp"
    HIDDEN4_MLP = "hidden4_mlp"
    HIDDEN1_CHANNELS_CNN = "hidden1_channels_cnn"
    HIDDEN2_CHANNELS_CNN = "hidden2_channels_cnn"
    HIDDEN3_CHANNELS_CNN = "hidden3_channels_cnn"
    HIDDEN4_CHANNELS_CNN = "hidden4_channels_cnn"
    HIDDEN5_CHANNELS_CNN = "hidden5_channels_cnn"
    KERNEL1_SIZE = "kernel1_size"
    KERNEL2_SIZE = "kernel2_size"
    KERNEL3_SIZE = "kernel3_size"
    KERNEL4_SIZE = "kernel4_size"
    KERNEL5_SIZE = "kernel5_size"
    STRIDE = "stride"
    PADDING = "padding"
    AUDIO_SRCS = "audio_srcs"
    IS_ONED = "is_oned"
    IS_TRAINING = "is_training"

class Optimizers(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSprop"
    SGD = "SGD"

class Models(str, Enum):
    VAE = "VAE"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUDIO_FILES_DIR_TRAIN = 'train'
AUDIO_FILES_DIR_VAL = 'validation'
AUDIO_FILES_DIR_TEST = 'test'
STEMS = ["bass","drums","guitar","piano"]

