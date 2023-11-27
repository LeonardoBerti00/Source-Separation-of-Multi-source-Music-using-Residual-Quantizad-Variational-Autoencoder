from einops import rearrange
from torch import nn
from utils.utils import compute_output_dim_convtranspose, compute_output_dim_conv
from models.ResidualLayer import ResidualLayer

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
                 num_convs: int,
                 ):
        super().__init__()
        """
        This is a convolutional neural network (CNN) used as encoder.
        """
        modules = []
        for i in range(num_convs):
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
        ##print(x.shape)
        x = self.Convs(x)
        ##print(x.shape)
        x = rearrange(x, 'b c h w -> b w (c h)')
        ##print(x.shape)
        x, _ = self.LSTM(x)
        ##print(x.shape)
        return x


class Encoder_CNN1D(nn.Module):
    """
    This is a convolutional neural network (CNN) used as encoder and decoder.
    """
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
                 num_convs: int,
                 is_residual: bool,
                 ):
        super().__init__()

        modules = []
        for i in range(num_convs):
            #print()
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels[i],
                          out_channels=hidden_channels[i + 1],
                          kernel_size=kernel_sizes[i][1],
                          stride=strides[i][1],
                          dilation=dilations[i][1],
                          padding=paddings[i][1],
                          bias=False
                          ),
                nn.BatchNorm1d(hidden_channels[i + 1]),
                nn.LeakyReLU(True),
                ResidualLayer(hidden_channels[i + 1]) if is_residual else None,
                )
            )
            if i == 0:
                output_dim = compute_output_dim_conv(input_dim=input_size,
                                                     kernel_size=kernel_sizes[i][1],
                                                     padding=paddings[i][1],
                                                     dilation=dilations[i][1],
                                                     stride=strides[i][1])
            else:
                output_dim = compute_output_dim_conv(input_dim=output_dim,
                                                     kernel_size=kernel_sizes[i][1],
                                                     padding=paddings[i][1],
                                                     dilation=dilations[i][1],
                                                     stride=strides[i][1])
            #print(output_dim)

        final_channel = -(len(kernel_sizes) - num_convs + 1)
        self.LSTM = nn.LSTM(input_size=hidden_channels[final_channel], hidden_size=latent_dim, num_layers=lstm_layers, batch_first=True)
        self.Convs = nn.Sequential(*modules)
        self.input_size = input_size

    def forward(self, x):
        #print(f"\nstart encoding: {x.shape}")
        x = self.Convs(x)
        #print(f"\nshape after convs: {x.shape}")
        x = rearrange(x, 'b c l -> b l c')
        #print(f"\nshape after rearrange: {x.shape}")
        x, _ = self.LSTM(x)
        #print(f"\nshape after LSTM: {x.shape}")
        return x



