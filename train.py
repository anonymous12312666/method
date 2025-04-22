import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import Dataset
import Net
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError
import Granual_Ball

# === parament set ===
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
PATCH_SIZE = 128
STRIDE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "Checkpoint/model.pth"
train_dataset = Dataset.FusionDataset("Testset/MEF/Source_A", "Testset/MEF/Source_B", patch_size=PATCH_SIZE, stride=STRIDE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
model = Net.GNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
to_pil = ToPILImage()
to_tensor = ToTensor()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
mse_metric = MeanSquaredError().to(DEVICE)

def sobel_filter(img):
    sobel_x = torch.tensor([[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]], dtype=torch.float32, device=img.device)
    sobel_y = torch.tensor([[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]], dtype=torch.float32, device=img.device)
    sobel_x = sobel_x.unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad


def laplacian_filter(img):
    lap_kernel = torch.tensor([[[[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]]]], dtype=torch.float32, device=img.device)
    return F.conv2d(img, lap_kernel, padding=1)

# === train===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if epoch % 2 == 0:
            x,y=y,x
        # == Build Supervise image ==
        target_list = []
        Q_list = []
        W_list = []

        for i in range(x.size(0)):
            pil_x = to_pil(x[i].cpu())
            pil_y = to_pil(y[i].cpu())
            # == Granual ball  ==
            fused_pil, q_ratio, w_ratio = Granual_Ball.image_fusion_algorithm(pil_x, pil_y)
            target_tensor = to_tensor(fused_pil)
            target_list.append(target_tensor)
            Q_list.append(q_ratio)
            W_list.append(w_ratio)

        target = torch.stack(target_list).to(DEVICE)
        BND_Ratio = torch.tensor(Q_list, device=DEVICE).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        SP_Ratio = torch.tensor(W_list, device=DEVICE).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # == forward ==
        output = model(x, y).to(DEVICE)


        # == Sobel Loss==
        sobel_output = sobel_filter(output)
        sobel_target = sobel_filter(target)
        sobel_x = sobel_filter(x)
        sobel_y = sobel_filter(y)
        # == NSP and SP guide==
        sobel_all = SP_Ratio* sobel_target +BND_Ratio* (sobel_x + sobel_y)
        sobel_loss = F.l1_loss(sobel_output, sobel_all, reduction='none')
        sobel_loss=sobel_loss.mean()

        # == Laplacian Loss ==
        lap_x = laplacian_filter(x)
        lap_y = laplacian_filter(y)
        lap_tar = laplacian_filter(target)
        loss_x_pout = F.l1_loss(lap_tar, lap_x)
        loss_y_pout = F.l1_loss(lap_tar, lap_y)

        # == SSIM Loss ==
        ssim_value_1 = ssim_metric(output, target).to(DEVICE)

        # == Total loss ==
        loss=(1-ssim_value_1)+loss_y_pout+loss_x_pout+1*sobel_loss

        # == ==
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"[{epoch+1}/{EPOCHS}] Avg Loss: {total_loss / len(train_loader):.6f}")

# === save model  ===
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model have saved to {SAVE_PATH}")
