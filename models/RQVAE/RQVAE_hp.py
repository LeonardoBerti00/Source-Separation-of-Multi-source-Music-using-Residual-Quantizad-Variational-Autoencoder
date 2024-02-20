import constants as cst

HP_RQVAE = {
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.001]},
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: {'values': [2048, 4096]},
    cst.LearningHyperParameter.NUM_QUANTIZERS.value: {'values': [4, 8]},
}

HP_RQVAE_FIXED = {
    cst.LearningHyperParameter.BATCH_SIZE.value: 16,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: 4096,
    cst.LearningHyperParameter.NUM_QUANTIZERS.value: 8,
}