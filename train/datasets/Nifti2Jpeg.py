import argparse
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path

# parse arg, enable usage like "Nifti2Jpeg ./input_dir -o ./output_dir"
parser = argparse.ArgumentParser(description='Convert Nifti to Jpeg')
parser.add_argument('-i', '--input_dir', type=str, help='input directory', default='./BraTS2021_Training_Data')
parser.add_argument('-o', '--output_dir', type=str, help='output directory', default='./nii2jpg_output')
parser.add_argument('-f', '--frames', type=int, help='frame index', default=60)

def read_nifti(nii_path: Path, frame: int):
    data = nib.load(nii_path).get_fdata()[:,:,frame]
    d_max = data.max()
    d_min = data.min()
    if d_max == d_min:
        return Image.fromarray(data.astype(np.uint8))

    # normalization
    data = (data - data.min()) / (data.max() - data.min())
    data *= 255

    return Image.fromarray(data.astype(np.uint8))


def recursive_convert(input: Path, output: Path, frame: int):
    for path in input.iterdir():
        if path.is_dir():
            mid = output.joinpath(path.name)
            if not mid.is_dir(): mid.mkdir()
            recursive_convert(path, mid, frame)
        if str(path).endswith(".nii.gz"):
            img = read_nifti(path, frame)
            img.save(output.joinpath(f"{path.stem}.jpg"), quality=100)


if __name__ == "__main__":
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frames = args.frames

    if not output_dir.exists(): output_dir.mkdir()
    if not input_dir.exists():
        raise FileNotFoundError("input or output directory does not exist")
    if not input_dir.is_dir() or not output_dir.is_dir():
        raise NotADirectoryError("input or output directory is not a directory")
    if frames < 50: raise ValueError("frame index should be greater than 50")
    if frames > 155: raise ValueError("frame index should be less than 155")

    recursive_convert(input_dir, output_dir, frames)

