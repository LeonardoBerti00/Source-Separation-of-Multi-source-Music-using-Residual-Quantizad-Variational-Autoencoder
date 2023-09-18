from einops import rearrange
from torch import nn
from utils import compute_output_dim_convtranspose, compute_output_dim_conv

class Decoder_CNN2D(nn.Module):
    def __init__(self,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 latent_dim: int,
                 lstm_layers: int,
                 batch_size: int,
                 emb_sample_len: int,
                 ):
        super().__init__()

        """
        Parameters:
        - hidden_channels: list, the number of channels in the hidden layers
        - kernel_sizes: list, the size of the kernels
        - latent_dim: int, the size of the latent dimension
        - lstm_layers: int, the number of lstm layers
        - batch_size: int, the size of the batch
        """

        self.LSTM = nn.LSTM(input_size=latent_dim, hidden_size=hidden_channels[-1], num_layers=lstm_layers, batch_first=True)
        #print(latent_dim)
        modules = []
        for i in range(1, len(kernel_sizes)+1):
            if i==1:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=emb_sample_len,
                                                                kernel_size=kernel_sizes[-i][1],
                                                                stride=strides[-i][1],
                                                                dilation=dilations[-i][1],
                                                                padding=paddings[-i][1]
                                                                )
            else:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=after_conv_sample_len,
                                                                         kernel_size=kernel_sizes[-i][1],
                                                                         stride=strides[-i][1],
                                                                         dilation=dilations[-i][1],
                                                                         padding=paddings[-i][1]
                                                                         )
            modules.append(nn.Sequential(
                #nn.ZeroPad2d((kernel_sizes[-i][1]//2, kernel_sizes[-i][1]//2, 0, 0)),
                nn.ConvTranspose2d(in_channels=hidden_channels[-i],
                                   out_channels=hidden_channels[-i-1],
                                   kernel_size=kernel_sizes[-i],
                                   stride=strides[-i],
                                   dilation=dilations[-i],
                                   padding=paddings[-i],
                                   bias=False
                                   ),
                nn.LeakyReLU(True))
            )
            #print(after_conv_sample_len)

        self.Convs = nn.Sequential(*modules)
        self.batch_size = batch_size

    def forward(self, x):

        #print("decoder")
        #print(x.shape)
        x, _ = self.LSTM(x)
        #print(x.shape)
        x = rearrange(x, 'b w c -> b c 1 w')
        #print(x.shape)
        x = self.Convs(x)
        #print(x.shape)

        return x


class Decoder_CNN1D(nn.Module):
    def __init__(self,
                 audio_srcs: int,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 latent_dim: int,
                 lstm_layers: int,
                 batch_size: int,
                 emb_sample_len: int,
                 ):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder and decoder.

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
        self.LSTM = nn.LSTM(input_size=latent_dim, hidden_size=hidden_channels[-1], num_layers=lstm_layers, batch_first=True)
        modules = []
        for i in range(1, len(kernel_sizes)+1):
            if i==1:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=emb_sample_len,
                                                                kernel_size=kernel_sizes[-i][1],
                                                                stride=strides[-i][1],
                                                                dilation=dilations[-i][1],
                                                                padding=paddings[-i][1]
                                                                )
                modules.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=hidden_channels[-i - 1],
                                       kernel_size=kernel_sizes[-i][1],
                                       stride=strides[-i][1],
                                       dilation=dilations[-i][1],
                                       padding=paddings[-i][1],
                                       bias=False
                                       ),
                    nn.LeakyReLU(True))
                )
            elif i==len(kernel_sizes):
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=after_conv_sample_len,
                                                                         kernel_size=kernel_sizes[-i][1],
                                                                         stride=strides[-i][1],
                                                                         dilation=dilations[-i][1],
                                                                         padding=paddings[-i][1]
                                                                         )
                modules.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=audio_srcs,
                                       kernel_size=kernel_sizes[-i][1],
                                       stride=strides[-i][1],
                                       dilation=dilations[-i][1],
                                       padding=paddings[-i][1],
                                       bias=False
                                       ),
                    nn.LeakyReLU(True))
                )
            else:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=after_conv_sample_len,
                                                                         kernel_size=kernel_sizes[-i][1],
                                                                         stride=strides[-i][1],
                                                                         dilation=dilations[-i][1],
                                                                         padding=paddings[-i][1]
                                                                         )
                modules.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=hidden_channels[-i-1],
                                       kernel_size=kernel_sizes[-i][1],
                                       stride=strides[-i][1],
                                       dilation=dilations[-i][1],
                                       padding=paddings[-i][1],
                                       bias=False
                                       ),
                    nn.LeakyReLU(True))
                )
            print(after_conv_sample_len)

        self.Convs = nn.Sequential(*modules)
        self.batch_size = batch_size


    def forward(self, x):
        print("decoder")
        print(x.shape)
        x, _ = self.LSTM(x)
        print(x.shape)
        x = rearrange(x, 'b w c -> b c w')
        x = self.Convs(x)
        print(x.shape)

        return x


class Decoder_MLP(nn.Module):
    def __init__(self, input_size: int, audio_srcs: int, hidden_dims: list):
        super().__init__()
        """
        This is a multilayer perceptron (MLP) used as encoder and decoder.

        Parameters:
        - input_size: int, the size of the input dimension
        - audio_srcs: int, the number of channels in the input
        - hidden_dims: list, the list of the hidden dimensions
        """
        modules = []
        for i in range(1, len(hidden_dims)+1):
            if i == len(hidden_dims):
                modules.append(nn.Sequential(
                    nn.Linear(hidden_dims[-i], input_size * audio_srcs),
                    nn.LeakyReLU(True)
                ))
            else:
                modules.append(nn.Sequential(
                    nn.Linear(hidden_dims[-i], hidden_dims[-i-1]),
                    nn.LeakyReLU(True)
                ))

        self.fcs = nn.Sequential(*modules)
        self.audio_srcs = audio_srcs

    def forward(self, x):
        x = self.fcs(x)
        x = rearrange(x, 'b (d c) -> b c d', c=self.audio_srcs)
        return x