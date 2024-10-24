from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import nibabel as nib
import logging


class BraTSDataset(Dataset):
    IMG_TYPE = ("flair", "t1", "t1ce", "t2")
    MASK_TYPE = "seg"

    def __init__(self, dataset_dir: str, key_frames: list[int]):
        data_path = Path(dataset_dir)

        self.key_frames = key_frames
        self.key_frames_len = len(key_frames)
        self.paths = [d for d in data_path.iterdir() if not d.name.startswith('.')]

        frame_size = self._frame_size()

        for frame in key_frames:
            if frame not in range(frame_size):
                raise ValueError(f"Frame {frame} is out of range")

    def _get_img(self, img_id: int, frame: int, img_type: str) -> np.ndarray:
        img_path: Path = self.paths[img_id]
        typed_img_path = img_path.joinpath(f"{img_path.name}_{img_type}.nii.gz")
        if not typed_img_path.exists():
            raise FileNotFoundError(f"{typed_img_path} does not exist")

        img = nib.load(typed_img_path).get_fdata()[:, :, frame]
        if img_type == self.MASK_TYPE:
            img = (img == 1)
            return img

        px_max = np.max(img)
        if px_max > 0: img = img / px_max

        return img.astype(np.float32)

    def _frame_size(self):
        img_path: Path = self.paths[0]
        typed_img_path = img_path.joinpath(f"{img_path.name}_{self.MASK_TYPE}.nii.gz")
        if not typed_img_path.exists():
            raise FileNotFoundError(f"{typed_img_path} does not exist")
        return nib.load(typed_img_path).get_fdata().shape[2]

    def __len__(self):
        # 有len(self.paths)个人，每人有len(key_frames)个帧
        return len(self.paths) * len(self.key_frames)

    def __getitem__(self, idx):
        id, frame_idx = divmod(idx, self.key_frames_len)
        frame = self.key_frames[frame_idx]

        image = np.array([self._get_img(id, frame, type) for type in self.IMG_TYPE])
        mask = self._get_img(id, frame, self.MASK_TYPE)

        image = torch.as_tensor(image.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()
        return {'image': image, 'mask': mask}

