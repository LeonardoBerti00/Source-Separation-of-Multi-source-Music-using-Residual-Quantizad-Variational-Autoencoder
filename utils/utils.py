

def compute_output_dim_conv(input_dim, kernel_size, padding, dilation, stride):
    return ((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

def compute_output_dim_convtranspose(input_dim, kernel_size, padding, dilation, stride):
    return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1




