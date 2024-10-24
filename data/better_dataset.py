from pathlib import Path
import numpy as np
import nibabel as nib
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
            self.images = np.load(self.data_path)["images"]
            self.masks = np.load(self.data_path)["masks"]
            logging.info(f"Loaded data with shape: {self.images.shape}")
        except FileNotFoundError:
            logging.error(f"{self.data_path} does not exist")
            exit(1)

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, idx):
        image = torch.as_tensor(self.images[idx].copy()).float().contiguous()
        mask = torch.as_tensor(self.masks[idx].copy()).long().contiguous()
        return { 'image': image, 'mask': mask }
