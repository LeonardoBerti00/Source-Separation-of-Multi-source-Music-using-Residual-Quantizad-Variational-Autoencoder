# Autoregressive Generation and Source Separation of Multi-source Raw Music using Residual Quantization

In this project you can find an implementation of an autoregressive transformer-based generative model, trained with the discrete codes produced by a neural audio codec model based on the residual quantized variational autoencoder  architecture. The model can generate new music and separate audio sources, making a step toward a general audio model. The model is trained on the Slakh dataset. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires Python and Conda. If you don't have them installed, please do so first.

### Preparing Data
This step is optional and required only if you are interested in training a model from scratch or reproducing the results.
Detailed instructions to download the Slakh dataset can be found [here](https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md).

### Installing

To set up the environment for this project, follow these steps:

1. Clone the repository:
```sh
git clone <repository_url>
```
2. Navigate to the project directory
3. Create a new Conda environment using the environment.yml file:
```sh
conda env create -f environment.yml
```
4. Activate the new Conda environment:
```sh
conda activate torch
```

## Usage
This section of the README walks through how to train and sample from a model. You can decide to train, test or sampling both from the RQ-VAE and the RQTransformer. The first is used to separate sources and the second to generate new music.

### Train
To start the training of the RQ-VAE, go to config.py and set to True config.IS_TRAINING and config.IS_TRAINING_AE and then use the following command:
```sh
python main.py
```
To train the RQTransformer you need to download the checkpoint the RQ-VAE and set to False config.IS_TRAINING_AE and then run the following command:
```sh
python main.py
```

The training is integrate with wandb, so if you want to have all the useful information in your wandb account, you need to set the key for the login and the project name in run.py. Finally you have to set to True config.IS_WANDB.

### Test
To test the RQ-VAE, so the source separation perfomance, you can download the pretrained model [here](https://drive.google.com/drive/folders/1TYOg-voDWhmwX7JkGNscoLnuCSm8-Kkj).
To test the RQTRansformer you can download the pretrained model [here](https://drive.google.com/file/d/11WHzxSWo3FAlYICXYc1Pdnw8PKBJerC-/view?usp=sharing). Note that to test the RQTransformer you need both models, because the RQTransformer uses the quantized representations of the RQ-VAE.
After you have installed the files you need to unzip them and move them to data/checkpoints/RQ-VAE or data/checkpoints/RQTransformer.

To start the testing of the RQ-VAE, go to config.py and set to True only config.IS_TESTING and config.IS_TRAINING_AE and then use the following command:
```sh
python main.py
```
To start the testing of the RQTransformer, go to config.py and set to True only config.IS_TESTING and then use the following command:
```sh
python main.py
```

### Separation 
The checkpoint provided of the RQ-VAE reach a test SI-SDRi (scale-invariant
signal to distortion ratio improvement) of 11.4916 in the source separation task.
In data/separations you can find some examples of separation, under the name of recon_* and the corresponding original source.

### Generation
If you want to generate some new music you can set to True config.IS_SAMPLING and run the following command
```sh
python main.py
```
Note that at the moment the performance of the RQTransformer are not so good so the music generated won't be pleasant to listen to, if you have some ideas for improvement please let me know.

### Other implementations
VQ-VAE and VAE are also implemented. For the VQ-VAE the best test SI-SDRi reached is 6.8866, for VAE is -3.2843. As expected they both underperform with respect of RQ-VAE.

### Report 
If you are curious and want to know more details on the tasks, the models architecture and the experiments you can read a short [report](https://drive.google.com/file/d/1-fbV_F0Yo43-3_ZILB10NjSp2UZBfLBF/view?usp=sharing).