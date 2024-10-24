from PIL import Image
import matplotlib.pyplot as plt
import io
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from model.unet_model import UNet


class ModelUNet:
    device = None
    model = None
    ckpt = None
    images = None
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using {self.device}")
        self.model = UNet(n_channels=4, n_classes=1)

    def load_ckpt(self):
        # search for the latest checkpoint in ./ckpt
        ckpt = Path("./ckpt").rglob("*.pth")
        if not ckpt:
            raise FileNotFoundError("No checkpoint found in ./ckpt")
        self.ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        # self.model = torch.load(self.ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device, weights_only=False))


    def predict(self, flair, t1, t1ce, t2, threshold: float = 0.75):
        images = [flair, t1, t1ce, t2]
        # 转换成4*240*240
        for i in range(len(images)):
            images[i] = Image.open(images[i])
            images[i] = images[i].convert("L")
            # images[i] = images[i].resize((240, 240))
            images[i] = np.array(images[i], dtype=np.float32)
            images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
            # images[i] = np.expand_dims(images[i], axis=-1)
        images = np.array([images])

        print(f"[INFO] Start prediction")
        self.model.eval()
        with torch.no_grad():
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            self.model.to(device=self.device)
            res = self.model(images)
        pred_mask = res.cpu().squeeze().detach().numpy()
        pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
        pred_mask = (pred_mask > (threshold * pred_mask.max())) * 255
        print(f"[INFO] Mask sum: {pred_mask.sum()}")
        if pred_mask.sum() > (1000 * 255): pred_mask = np.zeros(pred_mask.shape)
        pred_mask = pred_mask.astype(np.int8)

        # convert to PIL image
        im = Image.fromarray(pred_mask, mode="L")
        im.convert('RGB')
        img_byte_arr = io.BytesIO()
        # im.save(img_byte_arr, format='JPEG', quality=100)
        im.save(img_byte_arr, format='PNG', quality=100)
        img_byte_arr.seek(0)

        print(f"[INFO] Prediction done")
        return img_byte_arr


#  Test run model
if __name__ == "__main__":
    model = ModelUNet()
    model.load_ckpt()
    IMG_TYPE = ("flair", "t1", "t1ce", "t2")
    images = []
    for img_type in IMG_TYPE:
        img = Image.open(f"nii2jpg_output\BraTS2021_00621_{img_type}.nii.jpg")
        # img = f"train/datasets/nii2jpg_output/test2/BraTS2021_00621_{img_type}.nii.png"
        images.append(img)
    res = model.predict(*images)
    res.show()
    # images = model.load_data(Path("./data"), "img")
    # pred_mask = model.predict(images)
    # print(pred_mask)