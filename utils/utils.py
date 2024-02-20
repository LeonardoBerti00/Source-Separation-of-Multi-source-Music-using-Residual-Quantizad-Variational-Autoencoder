from pathlib import Path
from torch.utils.data import DataLoader
import torch
import constants as cst



def compute_final_output_dim(input_dim, kernel_sizes, paddings, dilations, strides, num_convs):
    for i in range(num_convs):
        if i == 0:
            emb_sample_len = compute_output_dim_conv(input_dim=input_dim,
                                                        kernel_size=kernel_sizes[i],
                                                        padding=paddings[i],
                                                        dilation=dilations[i],
                                                        stride=strides[i])
        else:
            emb_sample_len = compute_output_dim_conv(input_dim=emb_sample_len,
                                                        kernel_size=kernel_sizes[i],
                                                        padding=paddings[i],
                                                        dilation=dilations[i],
                                                        stride=strides[i])
    return emb_sample_len


def compute_output_dim_convtranspose(input_dim, kernel_size, padding, dilation, stride):
    return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


def compute_output_dim_conv(input_dim, kernel_size, padding, dilation, stride):
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1


def compute_mean_std(train_set, batch_size, num_workers, shuffle):
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    mean = 0.0
    std = 0.0
    num_samples = 0

    for batch in train_dataloader:
        data = torch.sum(batch, dim=-2)
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        mean += torch.mean(data, dim=1).sum(0)
        std += torch.std(data, dim=1).sum(0)
        num_samples += batch_size
        if num_samples % 10000 == 0:
            print(num_samples)
    print("final num smaples: ", num_samples)
    mean /= num_samples
    std /= num_samples

    return mean, std


def save_audio(self, waveform, filename):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().detach().numpy()
    # Save as WAV
    #sf.write(cst.RECON_DIR+'/' + filename +'.wav', waveform, cst.SAMPLE_RATE)

