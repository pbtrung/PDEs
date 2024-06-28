import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py


class PDEDataset(Dataset):
    def __init__(self, h5_file, dset_name):
        with h5py.File(h5_file, "r") as hf:
            self.input_true = torch.from_numpy(hf[dset_name][:, :3]).float()
            self.u_true = torch.from_numpy(hf[dset_name][:, 3]).float().unsqueeze(1)

    def __len__(self):
        return len(self.input_true)

    def __getitem__(self, idx):
        return self.input_true[idx], self.u_true[idx]

    def get_inputs(self):
        return self.input_true

    def get_outputs(self):
        return self.u_true


class NN(nn.Module):
    def __init__(
        self, input_dim, hidden_layers, hidden_dim, output_dim, activation_fn=nn.Sigmoid()
    ):
        super(NN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
