import constants as cst

HP_VQVAE = {
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [128, 256]},
    cst.LearningHyperParameter.DROPOUT.value: {'values': [0, 0.1]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.0003, 0.003, 0.001]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value, cst.Optimizers.LION.value]},
    cst.LearningHyperParameter.LATENT_DIM.value: {'values': [32, 64, 128]},
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: {'values': [512, 1024]},
    cst.LearningHyperParameter.LSTM_LAYERS.value: {'values': [1, 2]},
    cst.LearningHyperParameter.DURATION.value: {'values': [1, 2]},
    cst.LearningHyperParameter.NUM_CONVS.value: {'values': [3, 4, 5]},
    cst.LearningHyperParameter.IS_RESIDUAL.value: {'values': [True, False]},
}

HP_VQVAE_FIXED = {
    cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.DROPOUT.value: 0,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0001,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LATENT_DIM.value: 32,
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: 512,
    cst.LearningHyperParameter.LSTM_LAYERS.value: 2,
    cst.LearningHyperParameter.DURATION.value: 1,
}