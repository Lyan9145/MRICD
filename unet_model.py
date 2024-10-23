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
        self.model = UNet(n_channels=4, n_classes=1)

    def load_ckpt(self):
        # search for the latest checkpoint in ./ckpt
        ckpt = Path("./ckpt").rglob("*.pth")
        if not ckpt:
            raise FileNotFoundError("No checkpoint found in ./ckpt")
        self.ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device, weights_only=False))


    # def load_data(img_path: Path, data_type: str, key_frame: int = 50):
    #     IMG_TYPE = ("flair", "t1", "t1ce", "t2")
    #     MASK_TYPE = "seg"

    #     def get_img(path: Path, name: str, frame: int = 50):
    #         img_serial_path = path.joinpath(f"{path.name}_{name}.nii.gz")
    #         if not img_serial_path.exists():
    #             raise FileNotFoundError(f"{img_serial_path} does not exist")
    #         img = nib.load(img_serial_path).get_fdata()[:, :, frame]
    #         px_max = np.max(img)
    #         if px_max > 0:
    #             img = img / px_max
    #         return img.astype(np.float32)

    #     if data_type == "mask":
    #         image = get_img(img_path, MASK_TYPE, key_frame)
    #         return image.astype(np.int32)
        
    #     images = []
    #     for img_type in IMG_TYPE:
    #         image = get_img(img_path, img_type, key_frame)
    #         images.append(image)
        
    #     return np.array(images)

    def predict(self, flair, t1, t1ce, t2, threshold: float = 0.68):
        images = [flair, t1, t1ce, t2]
        # 转换成4*240*240
        for i in range(len(images)):
            images[i] = images[i].resize((240, 240))
        images = np.array(images)

        self.model.eval()
        with torch.no_grad():
            images.to(device=self.device, dtype=torch.float32)
            self.model.to(device=self.device)
            res = self.model(images)
        pred_mask = res.cpu().squeeze().detach().numpy()
        pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
        pred_mask = pred_mask > (threshold * pred_mask.max())
        pred_mask = pred_mask.astype(np.int8)

        # convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        im = Image.open(buf).convert('RGBA')

        return im


#  Test run model
if __name__ == "__main__":
    model = ModelUNet()
    model.load_ckpt()
    # images = model.load_data(Path("./data"), "img")
    # pred_mask = model.predict(images)
    # print(pred_mask)