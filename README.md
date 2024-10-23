# MRICD

## MRI Cyber Doctor

### How to run MRICD server:

install requirements from `requirements_pc.txt` for cuda or `requirements_npu.txt` for accend device.

Run file: `server.py`


### Dataset:

Link: [BraTS2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

put folder `BraTS2021_Traning_Data` in `./data`


## Nii2Jpg Usage：

Download dataset

```
python ./train/datasets/Nifti2Jpeg.py ./data/BraTS2021_Training_Data/BraTS2021_xxxxx  
```

 `python ./train/datasets/Nifti2Jpeg.py -h ` for more usage.


## Refrence:

[UNet3plus_pth](https://github.com/avBuffer/UNet3plus_pth)
