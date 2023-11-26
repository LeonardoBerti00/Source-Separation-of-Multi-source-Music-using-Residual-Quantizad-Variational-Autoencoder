from einops import rearrange
from torch import nn
from utils.utils import compute_output_dim_convtranspose, compute_output_dim_conv
from models.ResidualLayer import ResidualLayer

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
                 num_convs: int,
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
        modules = []
        for i in range(1, num_convs+1):
            if i==1:
                if (emb_sample_len == 64):
                    dilation = (dilations[-i][0], dilations[-i][1]+1)
                else:
                    dilation = (dilations[-i][0], dilations[-i][1])
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=emb_sample_len,
                                                                kernel_size=kernel_sizes[-i][1],
                                                                stride=strides[-i][1],
                                                                dilation=dilation,
                                                                padding=paddings[-i][1]
                                                                )
                modules.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_channels[-i],
                                       out_channels=hidden_channels[-i-1],
                                       kernel_size=kernel_sizes[-i],
                                       stride=strides[-i],
                                       dilation=dilation,
                                       padding=paddings[-i],
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
                 num_convs: int,
                 is_residual: bool,
                 duration: int,
                 ):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder and decoder.
        """
        self.LSTM = nn.LSTM(input_size=latent_dim, hidden_size=hidden_channels[num_convs], num_layers=lstm_layers, batch_first=True)
        modules = []
        start = len(kernel_sizes) - num_convs + 1
        end = len(kernel_sizes) + 1
        for i in range(start, end):
            if i==start:
                if emb_sample_len == 64:
                    dilation = dilations[-i][1]+1
                else:
                    dilation = dilations[-i][1]
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=emb_sample_len,
                                                                kernel_size=kernel_sizes[-i][1],
                                                                stride=strides[-i][1],
                                                                dilation=dilation,
                                                                padding=paddings[-i][1]
                                                                )
                modules.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=hidden_channels[-i - 1],
                                       kernel_size=kernel_sizes[-i][1],
                                       stride=strides[-i][1],
                                       dilation=dilation,
                                       padding=paddings[-i][1],
                                       bias=False
                                       ),
                    nn.BatchNorm1d(hidden_channels[-i - 1]),
                    nn.LeakyReLU(True),
                    ResidualLayer(hidden_channels[-i - 1]) if is_residual else None,
                    )
                )
            elif i==end:
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
                    )
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
                    nn.BatchNorm1d(hidden_channels[-i - 1]),
                    nn.LeakyReLU(True),
                    ResidualLayer(hidden_channels[-i - 1]) if is_residual else None,
                    )
                )
            print(after_conv_sample_len)

        self.Convs = nn.Sequential(*modules)
        self.batch_size = batch_size


    def forward(self, x):
        print(f"\nstart decoding: {x.shape}")
        x, _ = self.LSTM(x)
        print(f"\nprint shape after lstm: {x.shape}")
        x = rearrange(x, 'b l c -> b c l')
        print(f"\nprint shape after rearrange: {x.shape}")
        x = self.Convs(x)
        print(f"\nprint shape after convs: {x.shape}")

        return x


