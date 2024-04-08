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
        self.modules1 = nn.Sequential(
            nn.BatchNorm1d(hidden_channels[0]),
            nn.Conv1d(in_channels=hidden_channels[0],
                      out_channels=hidden_channels[1],
                      kernel_size=kernel_sizes[0],
                      stride=strides[0],
                      dilation=dilations[0],
                      padding=paddings[0],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[1]),
        )
        self.modules2 = nn.Sequential(
            nn.BatchNorm1d(hidden_channels[1]),
            nn.Conv1d(in_channels=hidden_channels[1],
                      out_channels=hidden_channels[2],
                      kernel_size=kernel_sizes[1],
                      stride=strides[1],
                      dilation=dilations[1],
                      padding=paddings[1],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[2]),
        )
        self.modules3 = nn.Sequential(
            nn.BatchNorm1d(hidden_channels[2]),
            nn.Conv1d(in_channels=hidden_channels[2],
                      out_channels=hidden_channels[3],
                      kernel_size=kernel_sizes[2],
                      stride=strides[2],
                      dilation=dilations[2],
                      padding=paddings[2],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[3]),
        )
        self.modules4 = nn.Sequential(
            nn.BatchNorm1d(hidden_channels[3]),
            nn.Conv1d(in_channels=hidden_channels[3],
                      out_channels=hidden_channels[4],
                      kernel_size=kernel_sizes[3],
                      stride=strides[3],
                      dilation=dilations[3],
                      padding=paddings[3],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[4]),
        )
        self.modules5 = nn.Sequential(
            nn.BatchNorm1d(hidden_channels[4]),
            nn.Conv1d(in_channels=hidden_channels[4],
                      out_channels=hidden_channels[5],
                      kernel_size=kernel_sizes[4],
                      stride=strides[4],
                      dilation=dilations[4],
                      padding=paddings[4],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[5]),
        )
        self.LSTM = nn.LSTM(
            input_size=hidden_channels[num_convs], 
            hidden_size=hidden_channels[num_convs], 
            num_layers=lstm_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_channels[num_convs]*2, hidden_channels[num_convs])
        self.input_size = input_size

    def forward(self, x):
        #print(f"\nstart encoding: {x.shape}")
        res1 = self.modules1(x)
        res2 = self.modules2(res1)
        res3 = self.modules3(res2)
        res4 = self.modules4(res3)
        res5 = self.modules5(res4)
        out = rearrange(res5, 'b c l -> b l c')
        #print(f"\nshape after rearrange: {x.shape}")
        out, _ = self.LSTM(out)
        out = self.fc(out)
        #print(f"\nshape after LSTM: {x.shape}")
        return out, res1, res2, res3, res4, res5



