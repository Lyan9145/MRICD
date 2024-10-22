import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from model.UNet import UNet

class ModelUNet(UNet):
    device = None
    model = None
    ckpt = None
    images = None
    def __init__(self):
        super(ModelUNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_channels=4, n_classes=1)

    def load_ckpt(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt))


    def load_data(img_path: Path, data_type: str, key_frame: int = 50):
        IMG_TYPE = ("flair", "t1", "t1ce", "t2")
        MASK_TYPE = "seg"

        def get_img(path: Path, name: str, frame: int = 50):
            img_serial_path = path.joinpath(f"{path.name}_{name}.nii.gz")
            if not img_serial_path.exists():
                raise FileNotFoundError(f"{img_serial_path} does not exist")
            img = nib.load(img_serial_path).get_fdata()[:, :, frame]
            px_max = np.max(img)
            if px_max > 0:
                img = img / px_max
            return img.astype(np.float32)

        if data_type == "mask":
            image = get_img(img_path, MASK_TYPE, key_frame)
            return image.astype(np.int32)
        
        images = []
        for img_type in IMG_TYPE:
            image = get_img(img_path, img_type, key_frame)
            images.append(image)
        
        return np.array(images)

    def predict(self, images: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            images.to(device=self.device, dtype=torch.float32)
            self.model.to(device=self.device)
            pred_mask = self.model(images)
        pred_mask = pred_mask.squeeze().cpu().detach().numpy()
        pred_mask = (pred_mask - np.min(pred_mask)) * 1.2 / (np.max(pred_mask) - np.min(pred_mask))
        pred_mask = np.astype(np.int8)
        return pred_mask