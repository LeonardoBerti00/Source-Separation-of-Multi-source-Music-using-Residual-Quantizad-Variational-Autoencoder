from torch import nn, Tensor
from torch.nn.functional import leaky_relu

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
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
    

class ResidualBlockDilation(nn.Module):
    def __init__(self, in_channels: int):
        super(ResidualBlockDilation, self).__init__()
        self.resblock = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, dilation=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.LeakyReLU(True),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock2 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=9, dilation=3, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.LeakyReLU(True),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock3 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=27, dilation=9, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.LeakyReLU(True),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
    def forward(self, input: Tensor) -> Tensor:
        res1 = leaky_relu(input + self.resblock(input), inplace=True)
        res2 = leaky_relu(res1 + self.resblock2(res1), inplace=True)
        return leaky_relu(res2 + self.resblock3(res2), inplace=True)