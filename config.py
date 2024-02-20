from constants import LearningHyperParameter
import constants as cst

class Configuration:
    """ Represents the configuration file of the simulation, containing all variables of the simulation. """
    def __init__(self):

        self.IS_WANDB = False
        self.IS_SWEEP = False
        self.IS_TESTING = False
        self.IS_TRAINING =  True
        self.IS_TRAINING_AE = False
        self.IS_DEBUG = False

        self.SEED = 0

        self.CHOSEN_AE = cst.Autoencoders.RQVAE
        self.CHOSEN_TRANSFORMER = cst.Transformers.RQTRANSFORMER
        self.CHOSEN_MODEL = self.CHOSEN_AE if self.IS_TRAINING_AE else self.CHOSEN_TRANSFORMER

        self.SWEEP_METHOD = 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.EARLY_STOPPING_METRIC = None

        self.FILENAME_CKPT = None

        self.HYPER_PARAMETERS = {hp: None for hp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 16
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.001
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.CODEBOOK_LENGTH] = 2048
        self.HYPER_PARAMETERS[LearningHyperParameter.LSTM_LAYERS] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.INIT_KMEANS] = True
        self.HYPER_PARAMETERS[LearningHyperParameter.Z_SCORE] = True
        self.HYPER_PARAMETERS[LearningHyperParameter.CONV_SETUP] = 3
        self.HYPER_PARAMETERS[LearningHyperParameter.RES_TYPE] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_QUANTIZERS] = 8

        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES] = [[7, 3, 3, 3, 3, 4, 3], [3, 7, 7, 6, 5, 5, 5], [3, 7, 7, 6, 5, 5], [3, 7, 7, 7, 5]]
        self.HYPER_PARAMETERS[LearningHyperParameter.STRIDES] = [[1, 5, 5, 3, 3, 4, 3], [1, 5, 5, 4, 3, 3, 3], [1, 5, 5, 4, 3, 3], [1, 5, 5, 5, 3]]
        self.HYPER_PARAMETERS[LearningHyperParameter.PADDINGS] = [[3, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        self.HYPER_PARAMETERS[LearningHyperParameter.DILATIONS] = [[1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        self.HYPER_PARAMETERS[LearningHyperParameter.HIDDEN_CHANNELS] = [[1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 4, 8, 16, 32, 64, 128], [1, 8, 16, 32, 64, 128, 256], [1, 16, 32, 64, 128, 256]]
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_CONVS] = len(
            self.HYPER_PARAMETERS[LearningHyperParameter.KERNEL_SIZES][self.HYPER_PARAMETERS[LearningHyperParameter.CONV_SETUP]]
            )
        self.HYPER_PARAMETERS[LearningHyperParameter.MULTI_SPECTRAL_RECON_LOSS_WEIGHT] = 1e-6
        self.HYPER_PARAMETERS[LearningHyperParameter.COMMITMENT_LOSS_WEIGHT] = 0.1

        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_HEADS] = 8
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_TRANSFORMER_LAYERS] = 6