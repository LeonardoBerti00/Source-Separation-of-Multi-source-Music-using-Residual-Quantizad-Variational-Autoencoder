import constants as cst

HP_RQVAE = {
    cst.LearningHyperParameter.BATCH_SIZE.value: [8, 16, 32],
    cst.LearningHyperParameter.LEARNING_RATE.value: [0.0001, 0.001],
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: [512, 4096, 8192],
    cst.LearningHyperParameter.NUM_QUANTIZERS.value: [8, 12],
    cst.LearningHyperParameter.RECON_LOSS_WEIGHT: [1, 3],
    cst.LearningHyperParameter.SDR_LOSS_WEIGHT: [0.1, 0.5, 1],
}

# other hyperparameters tested were: 
# ResNet = ResBlock, ResNetDilation, ResNetDilation2
# latent_dim = 128, 256, 512
# final len = 294, 441 

HP_RQVAE_FIXED = {
    cst.LearningHyperParameter.BATCH_SIZE.value: 16,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.CODEBOOK_LENGTH.value: 4096,
    cst.LearningHyperParameter.NUM_QUANTIZERS.value: 12,
}