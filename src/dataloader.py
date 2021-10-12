"""
Source: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py
"""

## Necessary Packages
import os
import random

import torch
import numpy as np

from .utils import random_generator, MinMaxScaler, extract_time


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(data_path, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """

    ori_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i : i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


class TimeSeriesDataLoader:
    def __init__(self, data_name, seq_length, no: int = 10000, dim: int = 5) -> None:
        super().__init__()
        assert data_name in ["stock", "energy", "sine"]
        self.data_dir = os.path.join(os.path.dirname(__file__), "../data")
        if data_name == "stock":
            data = real_data_loading(
                os.path.join(self.data_dir, "stock_data.csv"), seq_length
            )
        elif data_name == "energy":
            data = real_data_loading(
                os.path.join(self.data_dir, "energy_data.csv"), seq_length
            )
        else:
            assert isinstance(no, int) and isinstance(dim, int)
            data = sine_data_generation(no, seq_length, dim)
        self.data, self.min_val, self.max_val = self.MinMaxScaler(np.asarray(data))
        self.num_obs, self.seq_len, self.dim = self.data.shape
        self.get_time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def MinMaxScaler(self, data):
        """Min-Max Normalizer.
        Different from the one in the utils.py

        Args:
            - data: raw data

        Returns:
            - norm_data: normalized data
            - min_val: minimum values (for renormalization)
            - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    def get_time(self):
        self.T, self.max_seq_len = extract_time(self.data)

    def get_z(self, batch_size, T_mb):
        if not isinstance(T_mb, list):
            T_mb = list(T_mb.numpy())
        T_mb = list(map(int, T_mb))
        return torch.from_numpy(
            np.asarray(
                random_generator(batch_size, self.dim, T_mb, self.max_seq_len),
                dtype=np.float32,
            )
        ).to(self.device)

    def get_x_t(self, batch_size):
        idx = [i for i in range(self.num_obs)]
        random.shuffle(idx)
        idx = idx[:batch_size]
        batch_data = np.take(np.array(self.data, dtype=np.float32), idx, axis=0)
        batch_data_T = np.take(np.array(self.T, dtype=np.float32), idx, axis=0)
        return (
            torch.from_numpy(batch_data).to(self.device),
            torch.from_numpy(
                batch_data_T
            ),  # T should a simple CPU tensor, otherwise error is thrown
        )

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index, :]),
            torch.from_numpy(self.T[index]),
            torch.from_numpy(self.Z[index, :]),
        )

    def __len__(self):
        return self.num_obs
