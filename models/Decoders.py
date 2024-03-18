from einops import rearrange
from torch import nn
from utils.utils import compute_output_dim_convtranspose, compute_output_dim_conv
from models.Residual import ConvNeXtBlock, ResidualBlock, ResidualNetworkDilation, ResidualNetworkDilation2


class Decoder(nn.Module):
    def __init__(self,
                 audio_srcs: int,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 lstm_layers: int,
                 batch_size: int,
                 emb_sample_len: int,
                 num_convs: int,
                 dropout: float,
                 ):
        super().__init__()
        """
        The Decoder class is a PyTorch module used for decoding the latent representations in an audio processing system.

        This class is part of an audio processing system and is responsible for decoding the latent representations 
        back into audio signals. It uses a series of convolutional layers, followed by LSTM layers to achieve this.
        """

        self.LSTM = nn.LSTM(
            input_size=hidden_channels[num_convs], 
            hidden_size=hidden_channels[num_convs], 
            num_layers=lstm_layers, 
            batch_first=True,
            dropout=dropout
            )
        modules = []
        start = len(kernel_sizes) - num_convs + 1
        end = len(kernel_sizes) + 1
        for i in range(start, end):
            if i==start:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=emb_sample_len,
                                                                kernel_size=kernel_sizes[-i],
                                                                stride=strides[-i],
                                                                dilation=dilations[-i],
                                                                padding=paddings[-i]
                                                                )
                modules.append(nn.Sequential(
                    nn.BatchNorm1d(hidden_channels[-i]),
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=hidden_channels[-i - 1],
                                       kernel_size=kernel_sizes[-i],
                                       stride=strides[-i],
                                       dilation=dilations[-i],
                                       padding=paddings[-i],
                                       bias=False
                                       ),
                    nn.PReLU(init=0.1),
                    nn.Dropout(dropout),
                    ResidualBlock(hidden_channels[-i - 1]),
                    ResidualBlock(hidden_channels[-i - 1]),
                    )
                )
            elif i==end-1:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=after_conv_sample_len,
                                                                         kernel_size=kernel_sizes[-i],
                                                                         stride=strides[-i],
                                                                         dilation=dilations[-i],
                                                                         padding=paddings[-i]
                                                                         )
                modules.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=audio_srcs,
                                       kernel_size=kernel_sizes[-i],
                                       stride=strides[-i],
                                       dilation=dilations[-i],
                                       padding=paddings[-i],
                                       bias=False
                                       ),
                    )
                )
            else:
                after_conv_sample_len = compute_output_dim_convtranspose(input_dim=after_conv_sample_len,
                                                                         kernel_size=kernel_sizes[-i],
                                                                         stride=strides[-i],
                                                                         dilation=dilations[-i],
                                                                         padding=paddings[-i]
                                                                         )
                modules.append(nn.Sequential(
                    nn.BatchNorm1d(hidden_channels[-i]),
                    nn.ConvTranspose1d(in_channels=hidden_channels[-i],
                                       out_channels=hidden_channels[-i-1],
                                       kernel_size=kernel_sizes[-i],
                                       stride=strides[-i],
                                       dilation=dilations[-i],
                                       padding=paddings[-i],
                                       bias=False
                                       ),
                    nn.PReLU(init=0.1),
                    nn.Dropout(dropout),
                    ResidualBlock(hidden_channels[-i - 1]),
                    ResidualBlock(hidden_channels[-i - 1]),
                    )
                )
            #print(after_conv_sample_len)
        
        self.Convs = nn.Sequential(*modules)
        self.batch_size = batch_size


    def forward(self, x):
        #print(f"\nstart decoding: {x.shape}")
        x, _ = self.LSTM(x)
        #print(f"\nshape after lstm: {x.shape}")
        x = rearrange(x, 'b l c -> b c l')
        #print(f"\nshape after rearrange: {x.shape}")
        x = self.Convs(x)
        #print(f"\nshape after convs: {x.shape}")
        return x


