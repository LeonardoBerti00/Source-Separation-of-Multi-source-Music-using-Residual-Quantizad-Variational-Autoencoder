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
        self.modules1 = nn.Sequential(
            nn.BatchNorm1d(hidden_channels[-1]),
            nn.ConvTranspose1d(in_channels=hidden_channels[-1],
                      out_channels=hidden_channels[-2],
                      kernel_size=kernel_sizes[-1],
                      stride=strides[-1],
                      dilation=dilations[-1],
                      padding=paddings[-1],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[-2]),
        )
        self.modules2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_channels[-2],
                      out_channels=hidden_channels[-3],
                      kernel_size=kernel_sizes[-2],
                      stride=strides[-2],
                      dilation=dilations[-2],
                      padding=paddings[-2],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[-3]),
        )
        self.modules3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_channels[-3],
                      out_channels=hidden_channels[-4],
                      kernel_size=kernel_sizes[-3],
                      stride=strides[-3],
                      dilation=dilations[-3],
                      padding=paddings[-3],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[-4]),
        )
        self.modules4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_channels[-4],
                      out_channels=hidden_channels[-5],
                      kernel_size=kernel_sizes[-4],
                      stride=strides[-4],
                      dilation=dilations[-4],
                      padding=paddings[-4],
                      bias=False
                      ),
            nn.PReLU(init=0.1),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels[-5]),
        )
        self.modules5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_channels[-5],
                      out_channels=audio_srcs,
                      kernel_size=kernel_sizes[-5],
                      stride=strides[-5],
                      dilation=dilations[-5],
                      padding=paddings[-5],
                      bias=False
                      ),
        )
        self.LSTM = nn.LSTM(
            input_size=hidden_channels[num_convs], 
            hidden_size=hidden_channels[num_convs], 
            num_layers=lstm_layers, 
            batch_first=True,
            dropout=dropout,
            bidirectional=True
            )
        self.fc = nn.Linear(in_features=hidden_channels[num_convs]*2, out_features=hidden_channels[num_convs])


    def forward(self, x, res1, res2, res3, res4, res5):
        #print(f"\nstart decoding: {x.shape}")
        x, _ = self.LSTM(x)
        x = self.fc(x)
        #print(f"\nshape after lstm: {x.shape}")
        x = rearrange(x, 'b l c -> b c l')
        x = x + res5
        #print(f"\nshape after rearrange: {x.shape}")
        x = self.modules1(x)
        x = x + res4
        x = self.modules2(x)
        x = x + res3
        x = self.modules3(x)
        x = x + res2
        x = self.modules4(x)
        x = x + res1
        x = self.modules5(x)
        #print(f"\nshape after convs: {x.shape}")
        return x


