import constants as cst

HP_RQTR = {
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.001]},
    cst.LearningHyperParameter.NUM_HEADS.value: {'values': [4, 8]},
    cst.LearningHyperParameter.NUM_TRANSFORMER_LAYERS.value: {'values': [4, 6]},
}

HP_RQTR_FIXED = {
    cst.LearningHyperParameter.BATCH_SIZE.value: 16,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.NUM_HEADS.value: 8,
    cst.LearningHyperParameter.NUM_TRANSFORMER_LAYERS.value: 6,
}