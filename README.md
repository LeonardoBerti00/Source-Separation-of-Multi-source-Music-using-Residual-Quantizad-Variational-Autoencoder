# Multi-source Masked Language Model for Audio
Multi-source masked language model for audio based on quantized representations

start encoding: torch.Size([128, 1, 44100])

8820
1764
588
194
64

shape after convs: torch.Size([128, 32, 64])

shape after rearrange: torch.Size([128, 64, 32])

shape after LSTM: torch.Size([128, 64, 32])

start decoding: torch.Size([128, 64, 32])

print shape after lstm: torch.Size([128, 64, 32])

print shape after rearrange: torch.Size([128, 32, 64])

64
194
588
1764
8820
44100

print shape after convs: torch.Size([128, 4, 44100])