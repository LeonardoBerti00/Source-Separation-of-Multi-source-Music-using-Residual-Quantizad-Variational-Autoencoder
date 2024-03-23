from einops import rearrange
from torch import nn
from utils.utils import compute_output_dim_convtranspose, compute_output_dim_conv
from models.Residual import ResidualBlock, ResidualNetworkDilation, ResidualNetworkDilation2, ConvNeXtBlock

class Encoder(nn.Module):
    """
    This is a convolutional neural network (CNN) used as encoder.
    """
    def __init__(self,
                 input_size: int,
                 hidden_channels: list,
                 kernel_sizes: list,
                 strides: list,
                 paddings: list,
                 dilations: list,
                 lstm_layers: int,
                 num_convs: int,
                 dropout: float,
                 ):
        super().__init__()
        modules = []
        for i in range(num_convs):
            #print()
            modules.append(nn.Sequential(
                nn.BatchNorm1d(hidden_channels[i]),
                nn.Conv1d(in_channels=hidden_channels[i],
                          out_channels=hidden_channels[i + 1],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          dilation=dilations[i],
                          padding=paddings[i],
                          bias=False
                          ),
                nn.PReLU(init=0.1),
                nn.Dropout(dropout),
                ResidualBlock(hidden_channels[i + 1]),
                )
            )
            '''
            if i == 0:
                output_dim = compute_output_dim_conv(input_dim=input_size,
                                                     kernel_size=kernel_sizes[i],
                                                     padding=paddings[i],
                                                     dilation=dilations[i],
                                                     stride=strides[i])
            else:
                output_dim = compute_output_dim_conv(input_dim=output_dim,
                                                     kernel_size=kernel_sizes[i],
                                                     padding=paddings[i],
                                                     dilation=dilations[i],
                                                     stride=strides[i])
            print(output_dim)
            '''
        self.LSTM = nn.LSTM(
            input_size=hidden_channels[num_convs], 
            hidden_size=hidden_channels[num_convs], 
            num_layers=lstm_layers, 
            batch_first=True, 
            dropout=dropout
        )

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



