from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging


class BraTSDataset(Dataset):
    def __init__(self, data_npz_dir: str, npz_key: str = 'data'):
        self.data_path = Path(data_npz_dir)
        # self.augmentation = augmentation
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        # self.scale = scale
        try:
            self.raw_data = np.load(self.data_path)[npz_key]
            self.raw_data = self.raw_data.reshape(-1, 5, 240, 240)
            logging.info(f"Loaded data with shape: {self.raw_data.shape}")
            self.images = self.raw_data[:, 1:, :, :]
            self.masks = self.raw_data[:, 0, :, :]
        except FileNotFoundError:
            logging.error(f"{self.data_path} does not exist")
            exit(1)

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        image = torch.as_tensor(self.images[idx].copy()).byte().contiguous()
        mask = torch.as_tensor(self.masks[idx].copy()).byte().contiguous()
        return { 'image': image, 'mask': mask }
