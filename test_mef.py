import os
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import Net  # Ensure GNet is defined inside GBnet.py or Net module
import color_2
import time
# ==== Configuration ====
print(" begin")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Checkpoint/MEF.pth"
FOLDER1 = "Testset/MEF/Source_A"  # Path to folder A
FOLDER2 = "Testset/MEF/Source_B"  # Path to folder B
OUTPUT_FOLDER = "Results/MEF"  # Output folder path
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== Load the model ====
model = Net.GNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully")

# ==== Get list of images ====
image_names = sorted(os.listdir(FOLDER1))

# ==== Batch image processing ====
to_tensor = ToTensor()
total_time = 0.0

for name in image_names:
    path1 = os.path.join(FOLDER1, name)
    path2 = os.path.join(FOLDER2, name)

    if not os.path.exists(path2):
        print(f" Matching image not found: {path2}")
        continue

    img1 = Image.open(path1).convert("L")
    img2 = Image.open(path2).convert("L")

    x1 = to_tensor(img1).unsqueeze(0).to(DEVICE)
    x2 = to_tensor(img2).unsqueeze(0).to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        fused = model(x1, x2)

    fused = torch.nan_to_num(fused, nan=0.5, posinf=1.0, neginf=0.0)
    fused = fused.clamp(0, 1)

    end_time = time.time()
    duration = end_time - start_time
    total_time += duration

    fused_img = ToPILImage()(fused.squeeze(0).cpu())
    out_path = os.path.join(OUTPUT_FOLDER, name.replace(".tif", ".png"))
    fused_img.save(out_path)

    print(f"Processed: {name} -> {out_path} | Time: {duration:.3f} seconds")


# Optional: Uncomment the following block to convert fused Y channel images to RGB
color_2.fuse_ycbcr_to_rgb(
    img1_dir=FOLDER1,
    img2_dir=FOLDER2,
    fused_y_dir=OUTPUT_FOLDER,
    output_dir=OUTPUT_FOLDER
)
print(" All images have been fused successfully")