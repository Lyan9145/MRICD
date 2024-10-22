from functools import partial

import numpy as np
from pathlib import Path
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm

data_dir = Path('./BraTS2021_Training_Data')
KEY_FRAME = 50

IMG_TYPE = ("flair", "t1", "t1ce", "t2")
MASK_TYPE = "seg"

def load_data(img_path: Path, data_type: str, key_frame: int = KEY_FRAME):
    def get_img(path: Path, name: str, frame: int = KEY_FRAME):
        img_serial_path = path.joinpath(f"{path.name}_{name}.nii.gz")
        if not img_serial_path.exists():
            raise FileNotFoundError(f"{img_serial_path} does not exist")

        img = nib.load(img_serial_path).get_fdata()[:, :, frame]
        px_max = np.max(img)
        if px_max > 0: img = img / px_max
        return img.astype(np.float32)

    if data_type == "mask":
        image = get_img(img_path, MASK_TYPE, key_frame)
        return image.astype(np.int32)

    images = []
    for img_type in IMG_TYPE:
        image = get_img(img_path, img_type, key_frame)
        images.append(image)

    return np.array(images)


def get_data_path(dataset_path: Path) -> list:
    return [d for d in dataset_path.iterdir() if not d.name.startswith('.')]

def save_preprocessed_data(dataset_dir: Path, data_npz_dir: str, key_frame_slice: list[int]):
    paths = get_data_path(dataset_dir)

    with Pool() as p:
        masks = []
        for frame in key_frame_slice:
            masks.extend(list(tqdm(
                p.imap(partial(load_data, data_type="mask", key_frame=frame), paths),
                total=len(paths)
            )))

    with Pool() as p:
        images = []
        for frame in key_frame_slice:
            images.extend(list(tqdm(
                p.imap(partial(load_data, data_type="img", key_frame=frame), paths),
                total=len(paths)
            )))

    masks = np.array(masks)
    images = np.array(images)

    np.savez_compressed(data_npz_dir, masks=masks, images=images)
    print(f"Data saved, with shape: masks: {masks.shape}, images: {images.shape}.")


# memory estimation:
# 240 * 240 * 5(modality) * 155(series) * 1251(nums) * 8(dtype) = 52 GBytes
if __name__ == "__main__":
    save_preprocessed_data(data_dir, "./big_data.npz", list(range(50, 52)))