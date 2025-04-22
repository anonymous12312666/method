import os
import numpy as np
from PIL import Image

def gray_to_rgb_with_reference(gray_image, reference_rgb_image, alpha=1):

    gray_image = np.array(gray_image).astype(np.float32)
    reference_rgb = np.array(reference_rgb_image).astype(np.float32)

    # 计算参考图的 Cb 和 Cr
    Cb_ref = 128 + (-0.168736 * reference_rgb[:, :, 0] -
                    0.331264 * reference_rgb[:, :, 1] +
                    0.5 * reference_rgb[:, :, 2])
    Cr_ref = 128 + (0.5 * reference_rgb[:, :, 0] -
                    0.418688 * reference_rgb[:, :, 1] -
                    0.081312 * reference_rgb[:, :, 2])

    # 色彩权重抑制
    Cb = 128 + alpha * (Cb_ref - 128)
    Cr = 128 + alpha * (Cr_ref - 128)
    Y = gray_image

    # YCbCr to RGB 转换
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    # 裁剪到合法范围并组合
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    rgb_image = np.stack([R, G, B], axis=-1)
    return rgb_image

def convert_gray_folder_to_rgb(gray_folder, reference_folder, output_folder, alpha=1):
    os.makedirs(output_folder, exist_ok=True)
    print("Begin ycrcb to RGB")
    gray_list = sorted([f for f in os.listdir(gray_folder) if f.lower().endswith(('.png', '.jpg'))])
    ref_list = sorted([f for f in os.listdir(reference_folder) if f.lower().endswith(('.png', '.jpg'))])

    for i, (gray_name, ref_name) in enumerate(zip(gray_list, ref_list)):
        gray_path = os.path.join(gray_folder, gray_name)
        reference_path = os.path.join(reference_folder, ref_name)
        output_path = os.path.join(output_folder, gray_name)

        gray_image = Image.open(gray_path).convert('L')
        reference_rgb_image = Image.open(reference_path).convert('RGB')

        rgb_result = gray_to_rgb_with_reference(gray_image, reference_rgb_image, alpha=alpha)
        Image.fromarray(rgb_result).save(output_path)
        print(f"[{i+1}/{len(gray_list)}] 保存: {output_path}")



