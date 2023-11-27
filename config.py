import torch
from constants import LearningHyperParameter
import constants as cst

class Configuration:
    """ Represents the configuration file of the simulation, containing all variables of the simulation. """
    def __init__(self):

        self.IS_WANDB = False
        self.IS_SWEEP = False
        self.IS_TESTING = False
        self.IS_TRAINING = True
        self.IS_DEBUG = True
        self.IS_TRAINING_AE = True

        self.SEED = 0

        self.CHOSEN_AE = cst.Autoencoders.VQVAE
        self.CHOSEN_TRANSFORMER = cst.Transformers.TRANSFORMER
        self.CHOSEN_MODEL = self.CHOSEN_AE if self.IS_TRAINING_AE else self.CHOSEN_TRANSFORMER

        self.SWEEP_METHOD = 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.EARLY_STOPPING_METRIC = None

        self.IS_ONED = True

        self.HYPER_PARAMETERS = {hp: None for hp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.0001
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.DURATION] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH] = 512
        self.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_RESIDUAL] = True

        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 50
        self.HYPER_PARAMETERS[LearningHyperParameter.MIN_DURATION] = 12           # it's the minimum duration of a track
        self.HYPER_PARAMETERS[LearningHyperParameter.MAX_DURATION] = 640
        self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_RATE] = 22050
        self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH] = self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_RATE] * self.HYPER_PARAMETERS[LearningHyperParameter.DURATION]
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES] = [(1, 3), (1, 3), (1, 3), (1, 5), (4, 3)]
        self.HYPER_PARAMETERS[LearningHyperParameter.STRIDES] = [(1, 5), (1, 5), (1, 3), (1, 3), (4, 3)]
        self.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS] = [(0, 0), (0, 0), (0, 1), (0, 0), (0, 1)]
        self.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS] = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS] = [1, 4, 8, 16, 32, 32]
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_CONVS] = 3
        self.HYPER_PARAMETERS[LearningHyperParameter.BETA] = 0.25
        self.HYPER_PARAMETERS[LearningHyperParameter.AUG_SHIFT] = None