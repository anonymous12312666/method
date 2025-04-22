import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2

class FusionDataset(Dataset):
    def __init__(self, img1_dir, img2_dir, patch_size=128, stride=64):
        self.img1_paths = sorted([os.path.join(img1_dir, f) for f in os.listdir(img1_dir)])
        self.img2_paths = sorted([os.path.join(img2_dir, f) for f in os.listdir(img2_dir)])
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self._create_patches()

    def _to_y_channel(self, pil_img):
        # PIL → RGB numpy
        rgb = np.array(pil_img)
        ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = ycbcr[:, :, 0]
        return Image.fromarray(y)

    def _create_patches(self):
        patches = []
        for p1, p2 in zip(self.img1_paths, self.img2_paths):
            img1 = self._to_y_channel(Image.open(p1).convert('RGB'))
            img2 = self._to_y_channel(Image.open(p2).convert('RGB'))
            img1 = TF.to_tensor(img1)  # [1,H,W]
            img2 = TF.to_tensor(img2)

            C, H, W = img1.shape
            for i in range(0, H - self.patch_size + 1, self.stride):
                for j in range(0, W - self.patch_size + 1, self.stride):
                    patch1 = img1[:, i:i+self.patch_size, j:j+self.patch_size]
                    patch2 = img2[:, i:i+self.patch_size, j:j+self.patch_size]
                    patches.append((patch1, patch2))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]
