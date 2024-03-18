from torch import nn, Tensor
from torch.nn.functional import leaky_relu

class ResidualBlock(nn.Module):
    ''' inspired by resnet '''
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
    def forward(self, input: Tensor) -> Tensor:
        return leaky_relu(input + self.resblock(input), inplace=True)
    

class ConvNeXtBlock(nn.Module):
    ''' inspired by ConvNeXt '''
    def __init__(self, in_channels: int):
        super(ConvNeXtBlock, self).__init__()
        self.convblock = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.Conv1d(in_channels, in_channels*4, kernel_size=1, bias=False),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels*4, in_channels, kernel_size=1, bias=False),
                                      )
    def forward(self, input: Tensor) -> Tensor:
        return input + self.convblock(input)


class ResidualNetworkDilation(nn.Module):
    ''' inspired by Soundstream '''
    def __init__(self, in_channels: int):
        super(ResidualNetworkDilation, self).__init__()
        self.resblock = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, dilation=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock2 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=9, dilation=3, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock3 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=27, dilation=9, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
    def forward(self, input: Tensor) -> Tensor:
        res1 = leaky_relu(input + self.resblock(input), inplace=True)
        res2 = leaky_relu(res1 + self.resblock2(res1), inplace=True)
        return leaky_relu(res2 + self.resblock3(res2), inplace=True)
    

class ResidualNetworkDilation2(nn.Module):
    ''' inspired by Jukebox '''
    def __init__(self, in_channels: int):
        super(ResidualNetworkDilation2, self).__init__()
        self.resblock = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, dilation=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock2 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=6, dilation=3, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock3 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=18, dilation=9, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
        self.resblock4 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=54, dilation=27, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      nn.PReLU(init=0.1),
                                      nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(in_channels),
                                      )
    def forward(self, input: Tensor) -> Tensor:
        res1 = leaky_relu(input + self.resblock(input), inplace=True)
        res2 = leaky_relu(res1 + self.resblock2(res1), inplace=True)
        res3 = leaky_relu(res2 + self.resblock3(res2), inplace=True)
        return leaky_relu(res3 + self.resblock4(res3), inplace=True)