## Tools and Scripts

### `Nifti2Image.py`

**Description:**
This script converts NIfTI files to JPEG or PNG images. It processes a directory of NIfTI files, extracts a specific frame from each file, normalizes the pixel values, and saves the resulting images in the specified output directory.

**Usage:**
```bash
python Nifti2Image.py --input_dir ./BraTS2021_Training_Data --output_dir ./nii2img_output2 --frames 60 --format png
```

### `data_preV2.py`

**Description:**
The improved version of `data_preV1.py`. Convert and generate the sliced dataset into a NPZ file, which is **NEEDED** when training model using **train_ram.py**. It preprocesses the BraTS2021 dataset by loading specific frames from NIfTI files, normalizing pixel values, and saving the preprocessed data as an NPZ file.

**Usage:**
```bash
python data_preV2.py
```

### `download_datasets.py`

**Description:**
This script downloads the BraTS2021 dataset from Kaggle using the `kagglehub` library. It prints the path to the downloaded dataset files.

**Usage:**
```bash
python download_datasets.py
```

### `glance.ipynb`

**Description:**
This Jupyter notebook provides a quick glance at the BraTS2021 dataset. It loads images and masks, visualizes them, and prints their shapes and data types.

## Directory Structure

```
data/
├── Nifti2Image.py
├── README.md
├── __init__.py
├── better_dataset.py
├── data_preV1.py
├── data_preV2.py
├── datasets/
├── diff.ipynb
├── download_datasets.py
├── glance.ipynb
├── lazy_dataset.py
├── nii2jpg_output/
└── origin_dataset.py
```
