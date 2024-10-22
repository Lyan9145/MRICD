import numpy as np
from pathlib import Path
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm

data_dir = Path('/gemini/data-1/BraTS2021/BraTS2021_Training_Data')
IMG_TYPE = ("seg", "flair", "t1", "t1ce", "t2")

def load_data(img_path: Path):
    ret = []
    for img_type in IMG_TYPE:
        img_serial_path = img_path.joinpath(f"{img_path.name}_{img_type}.nii.gz")
        if not img_serial_path.exists():
            raise FileNotFoundError(f"{img_serial_path} does not exist")

        img = nib.load(img_serial_path).get_fdata()
        for i in range(img.shape[2]):
            px_max = np.max(img[:, :, i])
            if px_max > 0:
                img[:, :, i] = img[:, :, i] / px_max
                img[:, :, i] *= 255
        img = img.astype(np.float16)
        ret.append(img)
    return np.transpose(np.array(ret), (3, 0, 1, 2))


def get_data_path(dataset_path: Path) -> list:
    return [d for d in dataset_path.iterdir() if not d.name.startswith('.')]

def save_preprocessed_data(dataset_dir: Path, data_npz_dir: str, data_size: int = 0):
    paths = get_data_path(dataset_dir)
    if data_size > 0: paths = paths[:data_size]

    with Pool() as p:
        data = list(tqdm(
            p.imap(load_data, paths),
            total=len(paths)
        ))

        data = np.array(data)
        np.savez_compressed(data_npz_dir, data=data)
        print(f"Data saved, with shape: {data.shape}.")


# memory estimation:
# 240 * 240 * 5(modality) * 155(series) * 1251(nums) * 8(dtype) = 52 GBytes
if __name__ == "__main__":
    save_preprocessed_data(data_dir, "./mini_data.npz", 50)