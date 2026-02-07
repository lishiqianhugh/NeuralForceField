import numpy as np
import torch
from torch.utils.data import Dataset
from utils.util import vis_nbody_traj

class NBodyDataset(Dataset):
    def __init__(self, file_path, sample_num=100):
        """
        Args:
            file_path (str): Path to the .npy file containing the data.
        """
        self.data = np.load(file_path, allow_pickle=True)[:sample_num]
        '''
        data [sample_num, steps=100, body_num=4, feature_dim=10]:
        0: mass
        1: x
        2: y
        3: z
        4: vx
        5: vy
        6: vz
        7: ax
        8: ay
        9: az
        '''

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Sample containing data for body_num, steps, and feature_dim.
        """
        sample = self.data[idx][...,:7]
        return torch.tensor(sample, dtype=torch.float32)

if __name__ == '__main__':
    dataset = NBodyDataset('nbody_data.npy')
    vis_nbody_traj(dataset[0].numpy())
