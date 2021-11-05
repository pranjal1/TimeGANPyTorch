import torch
import numpy as np


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0.0, 1, [T_mb[i], z_dim])
        temp[: T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def get_moments(d, n=2):
    assert n <= 4
    # Return the first n moments of the data provided
    mean = torch.mean(d)
    if n == 1:
        return mean.reshape(1,)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    if n == 2:
        return (
            mean.reshape(1,),
            std.reshape(1,),
        )
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    if n == 3:
        return (
            mean.reshape(1,),
            std.reshape(1,),
            skews.reshape(1,),
        )
    kurtoses = (
        torch.mean(torch.pow(zscores, 4.0)) - 3.0
    )  # excess kurtosis, should be 0 for Gaussian
    final = (
        mean.reshape(1,),
        std.reshape(1,),
        skews.reshape(1,),
        kurtoses.reshape(1,),
    )
    return final
