# Multi-source Masked Language Model for Audio
Multi-source masked language model for audio based on quantized representations

bat16_cod_4090_num_8
 test sdri on epoch 0 is 5.5826 with 10000

 test sdri 5.9 with 20000

 test sdri 6.11 with 80000


RQ-VAE without training codebook
  test sdri on epoch 0 is 6.1163

RQ-VAE re-init ema
  test sdri on epoch 0 is 7.1392

RQ-VAE 2048 re-init ema
  test sdri on epoch 0 is 6.6325

RQ-VAE re-init without ema 
  test sdri on epoch 0 is 6.4452

RQ-VAE 8 re-init ema
  test sdri on epoch 0 is 6.5268

RQ-VAE loss paper
  test sdri on epoch 0 is 6.1385

RQ-VAE noise
  test sdri on epoch 0 is 6.4929

VQ-VAE without training codebook:
  test sdri on epoch 7 is 6.6332

VQ-VAE training codebook:
  test sdri on epoch 0 is 6.8866