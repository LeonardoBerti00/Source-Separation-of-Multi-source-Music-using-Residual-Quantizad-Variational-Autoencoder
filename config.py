import torch
from constants import LearningHyperParameter
import constants as cst

class Configuration:
    """ Represents the configuration file of the simulation, containing all variables of the simulation. """
    def __init__(self):

        self.IS_TEST_ONLY = False

        self.SEED = 0

        self.CHOSEN_MODEL = cst.Models.VAE

        self.IS_WANDB = False

        self.SWEEP_METHOD = 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.SWEEP_METRIC = {
            'goal': 'minimize',
            'name': None
        }

        self.EARLY_STOPPING_METRIC = None

        self.HYPER_PARAMETERS = {hp: None for hp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.01
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.SGD.value
        self.HYPER_PARAMETERS[LearningHyperParameter.WEIGHT_DECAY] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.EPS] = 1e-08  # default value for ADAM
        self.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM] = 0.9
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_RATE] = 22050
        self.HYPER_PARAMETERS[LearningHyperParameter.DURATION] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_LENGTH] = self.HYPER_PARAMETERS[LearningHyperParameter.SAMPLE_RATE] * self.HYPER_PARAMETERS[LearningHyperParameter.DURATION]
        self.HYPER_PARAMETERS[LearningHyperParameter.CHANNEL_SIZE] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.AUG_SHIFT] = False
        self.HYPER_PARAMETERS[LearningHyperParameter.MIN_DURATION] = 12
        self.HYPER_PARAMETERS[LearningHyperParameter.MAX_DURATION] = 640
        self.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN1_MLP] = 4096
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN2_MLP] = 1024
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN3_MLP] = 256
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN4_MLP] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL1_SIZE] = (1,5)
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL2_SIZE] = (1,5)
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL3_SIZE] = (1,3)
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL4_SIZE] = (1,3)
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL5_SIZE] = (1,2)
        self.HYPER_PARAMETERS[LearningHyperParameter.STRIDE] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.PADDING] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN1_CHANNELS_CNN] = 8
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN2_CHANNELS_CNN] = 16
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN3_CHANNELS_CNN] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN4_CHANNELS_CNN] = 64
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN5_CHANNELS_CNN] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.AUDIO_SRCS] = 4
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_ONED] = False
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_TRAINING] = True

        self.NUM_WORKERS = 0
