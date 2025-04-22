import os
from PIL import Image
import numpy as np

def fuse_ycbcr_to_rgb(img1_dir, img2_dir, fused_y_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    print("Begin ycrcb to RGB")

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    img1_list = sorted([f for f in os.listdir(img1_dir) if f.lower().endswith(valid_exts)])
    img2_list = sorted([f for f in os.listdir(img2_dir) if f.lower().endswith(valid_exts)])
    fused_y_list = sorted([f for f in os.listdir(fused_y_dir) if f.lower().endswith(valid_exts)])

    for img1_name, img2_name, fused_y_name in zip(img1_list, img2_list, fused_y_list):
        img1 = Image.open(os.path.join(img1_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(img2_dir, img2_name)).convert('RGB')
        imgf_y = Image.open(os.path.join(fused_y_dir, fused_y_name)).convert('L')

        ycbcr1 = img1.convert('YCbCr')
        ycbcr2 = img2.convert('YCbCr')
        ycbcr1_np = np.array(ycbcr1).astype(np.float32)
        ycbcr2_np = np.array(ycbcr2).astype(np.float32)
        imgf_y_np = np.array(imgf_y).astype(np.float32)

        cb1, cr1 = ycbcr1_np[:, :, 1], ycbcr1_np[:, :, 2]
        cb2, cr2 = ycbcr2_np[:, :, 1], ycbcr2_np[:, :, 2]

        # 融合 Cb
        mask_same128 = (cb1 == 128) & (cb2 == 128)
        mask_diff = ~mask_same128
        numerator_cb = cb1 * np.abs(cb1 - 128) + cb2 * np.abs(cb2 - 128)
        denominator_cb = np.abs(cb1 - 128) + np.abs(cb2 - 128)
        cbf = np.zeros_like(cb1)
        cbf[mask_same128] = 128
        cbf[mask_diff] = numerator_cb[mask_diff] / (denominator_cb[mask_diff] + 1e-8)

        # 融合 Cr
        mask_same128 = (cr1 == 128) & (cr2 == 128)
        mask_diff = ~mask_same128
        numerator_cr = cr1 * np.abs(cr1 - 128) + cr2 * np.abs(cr2 - 128)
        denominator_cr = np.abs(cr1 - 128) + np.abs(cr2 - 128)
        crf = np.zeros_like(cr1)
        crf[mask_same128] = 128
        crf[mask_diff] = numerator_cr[mask_diff] / (denominator_cr[mask_diff] + 1e-8)

        fused_ycbcr = np.stack((imgf_y_np, cbf, crf), axis=2)
        fused_ycbcr = np.clip(fused_ycbcr, 0, 255).astype(np.uint8)
        fused_img = Image.fromarray(fused_ycbcr, mode='YCbCr').convert('RGB')

        base_name = os.path.splitext(fused_y_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.jpg")
        fused_img.save(output_path)
        print(f"saved：{output_path}")
