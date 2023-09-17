import torch
from constants import LearningHyperParameter
import constants as cst

class Configuration:
    """ Represents the configuration file of the simulation, containing all variables of the simulation. """
    def __init__(self):

        self.IS_TEST_ONLY = False

        self.SEED = 0

        self.CHOSEN_MODEL = cst.Models.VAE

        self.IS_SWEEP = False

        self.SWEEP_METHOD = 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.SWEEP_METRIC = {
            'goal': 'minimize',
            'name': None
        }

        self.EARLY_STOPPING_METRIC = None

        self.IS_ONED = False
        self.IS_TRAINING = True

        self.HYPER_PARAMETERS = {hp: None for hp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 64
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.00001
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value
        self.HYPER_PARAMETERS[LearningHyperParameter.WEIGHT_DECAY] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.EPS] = 1e-08  # default value for ADAM
        self.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM] = 0.9
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_RATE] = 22050
        self.HYPER_PARAMETERS[LearningHyperParameter.DURATION] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH] = self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_RATE] * self.HYPER_PARAMETERS[LearningHyperParameter.DURATION]
        self.HYPER_PARAMETERS[LearningHyperParameter.MIN_DURATION] = 12
        self.HYPER_PARAMETERS[LearningHyperParameter.MAX_DURATION] = 640
        self.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_MLP] = [4096, 1024, 256, 128]
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES] = [(1, 3), (1, 3), (1, 3), (1, 5), (4, 3)]
        self.HYPER_PARAMETERS[LearningHyperParameter.STRIDES] = [(1, 5), (1, 5), (1,3), (1,3), (4,3)]
        self.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS] = [(0, 0), (0, 0), (0, 1), (0, 0), (0, 1)]
        self.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS] = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        self.HYPER_PARAMETERS[LearningHyperParameter.CHANNEL_SIZE] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS] = [1, 8, 16, 32, 64, 128]
        self.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH] = 512
        self.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.COMMITMENT_COST] = 0.25

        self.NUM_WORKERS = 0

        self.AUDIO_SRCS = 4
        self.AUG_SHIFTS = False