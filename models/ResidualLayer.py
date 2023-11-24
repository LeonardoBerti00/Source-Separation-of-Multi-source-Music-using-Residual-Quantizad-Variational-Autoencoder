from torch import nn, Tensor
from torch.nn.functional import leaky_relu

class ResidualLayer(nn.Module):

    def __init__(self, in_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.LeakyReLU(True),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.LeakyReLU(True),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
    def forward(self, input: Tensor) -> Tensor:
        return leaky_relu(input + self.resblock(input), inplace=True)