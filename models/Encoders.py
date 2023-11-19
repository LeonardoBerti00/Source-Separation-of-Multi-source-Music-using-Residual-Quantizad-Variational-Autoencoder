from einops import rearrange
from torch import nn
from utils import compute_output_dim_convtranspose, compute_output_dim_conv


class Encoder_CNN2D(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 latent_dim: int,
                 lstm_layers: int,
                 ):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - channel_size: int, the number of channels in the input
        - hidden1: int, the size of the first hidden layer
        - hidden2: int, the size of the second hidden layer
        - hidden3: int, the size of the third hidden layer
        - hidden4: int, the size of the fourth hidden layer
        - kernel_size: int, the size of the kernel
        - latent_dim: int, the size of the latent dimension
        """
        modules = []
        for i in range(len(kernel_sizes)):
            # Last layer merges the four audio sources
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels[i],
                          out_channels=hidden_channels[i+1],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          dilation=dilations[i],
                          padding=paddings[i],
                          bias=False
                          ),
                nn.LeakyReLU(True),
                nn.BatchNorm2d(hidden_channels[i+1]))
            )

        self.LSTM = nn.LSTM(input_size=hidden_channels[-1], hidden_size=latent_dim, num_layers=lstm_layers, batch_first=True)
        self.Convs = nn.Sequential(*modules)
        self.input_size = input_size


    def forward(self, x):
        x = rearrange(x, 'b h w -> b 1 h w')
        #print(x.shape)
        x = self.Convs(x)

        x = rearrange(x, 'b c h w -> b w (c h)')
        #print(x.shape)
        x, _ = self.LSTM(x)
        #print(x.shape)
        return x


class Encoder_CNN1D(nn.Module):
    def __init__(self,
                 input_size: int,
                 audio_srcs: int,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 latent_dim: int,
                 lstm_layers: int,
                 ):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder and decoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - channel_size: int, the number of channels in the input
        - hidden_channels: list, the number of channels in the hidden layers
        - kernel_size: int, the size of the kernel
        - latent_dim: int, the size of the latent dimension
        """

        modules = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                modules.append(nn.Sequential(
                    nn.Conv1d(in_channels=audio_srcs,
                              out_channels=hidden_channels[i+1],
                              kernel_size=kernel_sizes[i][1],
                              stride=strides[i][1],
                              dilation=dilations[i][1],
                              padding=paddings[i][1],
                              bias=False
                              ),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(hidden_channels[i+1]))
                )
            else:
                modules.append(nn.Sequential(
                    nn.Conv1d(in_channels=hidden_channels[i],
                              out_channels=hidden_channels[i + 1],
                              kernel_size=kernel_sizes[i][1],
                              stride=strides[i][1],
                              dilation=dilations[i][1],
                              padding=paddings[i][1],
                              bias=False
                              ),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(hidden_channels[i + 1]))
                )

        self.LSTM = nn.LSTM(input_size=hidden_channels[-1], hidden_size=latent_dim, num_layers=lstm_layers, batch_first=True)
        self.Convs = nn.Sequential(*modules)
        self.input_size = input_size

    def forward(self, x):
        print(x.shape)
        x = self.Convs(x)
        print(x.shape)
        x = rearrange(x, 'b c w -> b w c')
        print(x.shape)
        x, _ = self.LSTM(x)
        print(x.shape)
        return x



