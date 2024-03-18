import constants as cst

HP_VQVAE = {
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.001]},
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: {'values': [512, 1024]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32]},
}

HP_VQVAE_FIXED = {
    cst.LearningHyperParameter.BATCH_SIZE.value: 16,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: 8092,
}