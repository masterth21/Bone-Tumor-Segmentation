import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from mkunet_network import MK_UNet
from utils.dataloader_bone import get_loader
from utils.utils import clip_gradient, AvgMeter, cal_params_flops

# Tự động chọn thiết bị: Ưu tiên GPU (cuda), nếu không có thì dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def structure_loss(pred, mask):
    """
    Hàm loss hỗn hợp 1:1 giữa Weighted BCE và Weighted IoU.
    Giúp mô hình học tốt biên giới u xương từ nhãn đa giác (polygon).
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    # Weighted Binary Cross Entropy
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted IoU
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_coefficient(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def iou_metric(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def test(model, path, split, opt):
    model.eval()
    image_root = os.path.join(path, split, "images/")
    gt_root = os.path.join(path, split, "masks/")

    test_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        split=split,
        color_image=opt.color_image,
    )

    DSC, IOU, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            # Code xử lý adaptive cho cả dataloader trả về 2 hoặc 4 giá trị
            images = pack[0].to(device)
            gts = pack[1].to(device).float()

            # Lấy thông tin shape để resize (nếu có)
            original_shapes = pack[2] if len(pack) > 2 else None

            ress = model(images)
            predictions = ress[0] if isinstance(ress, list) else ress

            for i in range(len(images)):
                p = predictions[i].unsqueeze(0)
                g = gts[i].unsqueeze(0)

                if original_shapes is not None:
                    h_orig, w_orig = int(original_shapes[0][i]), int(
                        original_shapes[1][i]
                    )
                    pred_resized = F.interpolate(
                        p, size=(h_orig, w_orig), mode="bilinear", align_corners=False
                    )
                    gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode="nearest")
                else:
                    pred_resized = p
                    gt_resized = g

                pred_binary = (pred_resized.sigmoid() >= 0.5).float()
                gt_binary = (gt_resized >= 0.5).float()

                total_images += 1
                DSC += dice_coefficient(pred_binary, gt_binary).item()
                IOU += iou_metric(pred_binary, gt_binary).item()

    return (DSC / total_images if total_images > 0 else 0), (
        IOU / total_images if total_images > 0 else 0
    )


def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()
    global best, total_train_time, dict_plot

    epoch_start = time.time()
    loss_record = AvgMeter()
    # Multi-scale training {0.75, 1.0, 1.25}
    size_rates = [0.75, 1, 1.25]
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = images.to(device), gts.float().to(device)

            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="nearest")

            out = model(images)
            loss = structure_loss(out[0] if isinstance(out, list) else out, gts)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:  # Tăng tần suất in log khi dùng CPU
            print(
                f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], "
                f"Loss: {loss_record.show():.4f}"
            )

    total_train_time += time.time() - epoch_start

    # Validation sau mỗi epoch
    val_dice, val_iou = test(model, opt.data_path, "val", opt)
    print(f"--- Epoch: {epoch} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} ---")

    if val_dice > best:
        best = val_dice
        save_path = opt.train_save
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth")
        )
        print(f">>> Best Model Saved (Dice: {best:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="MK_UNet")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batchsize", type=int, default=4)  # Giảm bs cho CPU
    parser.add_argument("--test_batchsize", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="./data/processed/")
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--color_image", default=True)
    parser.add_argument("--augmentation", default=False)
    opt = parser.parse_args()

    NET_CONFIGS = {"MK_UNet": [16, 32, 64, 96, 160]}

    best = 0.0
    total_train_time = 0

    # Khởi tạo mô hình trên Device đã chọn (CPU hoặc GPU)
    print(f"Using device: {device}")
    channels = NET_CONFIGS.get(opt.network, [16, 32, 64, 96, 160])
    model = MK_UNet(num_classes=1, in_channels=3, channels=channels).to(device)

    # Logs & Saves
    run_id = f"BoneTumor_{datetime.now().strftime('%m%d_%H%M')}"
    opt.train_save = f"./model_pth/{run_id}/"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename=f"logs/{run_id}.log", level=logging.INFO)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

    # Dataloader trỏ vào data/processed
    train_loader = get_loader(
        image_root=os.path.join(opt.data_path, "train/images/"),
        gt_root=os.path.join(opt.data_path, "train/masks/"),
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        augmentation=opt.augmentation,
        split="train",
    )

    print(f"Starting Training: {run_id}")

    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt, run_id)
        scheduler.step()

    print("\n--- Final Test Evaluation ---")
    test_dice, test_iou = test(model, opt.data_path, "test", opt)
    print(f"Final Test Results - Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")
