import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

# Project-specific imports
from mkunet_network import MK_UNet

# THAY ĐỔI: Sử dụng dataloader_bone thay vì dataloader_polyp
from utils.dataloader_bone import get_loader
from medpy.metric.binary import hd95

# --- THAY ĐỔI: Thiết lập Device tự động (Hỗ trợ chạy trên CPU nếu không có GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    try:
        if pred.sum() > 0 and gt.sum() > 0:
            hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy())
        else:
            hd_val = 100.0
    except:
        hd_val = 100.0

    return sensitivity, specificity, precision, hd_val


def test(model, path, split, opt, save_base=None):
    # THAY ĐỔI: Logic đường dẫn cho u xương (trỏ vào data/processed)
    image_root = os.path.join(path, split, "images/")
    gt_root = os.path.join(path, split, "masks/")
    model.eval()

    test_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        split=split,
        augmentation=False,  # Bone test không dùng Augmentation
    )

    DSC, IOU, total_images = 0.0, 0.0, 0
    detailed_results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f"Inference on {split}"):
            # THAY ĐỔI: Hỗ trợ linh hoạt dataloader trả về 4 giá trị
            images = pack[0].to(device)
            gts = pack[1].to(device).float()
            original_shapes = pack[2]
            names = pack[3]

            ress = model(images)
            predictions = ress[0] if isinstance(ress, list) else ress

            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])

                p = predictions[i].unsqueeze(0)
                pred_resized = (
                    F.interpolate(
                        p, size=(h_orig, w_orig), mode="bilinear", align_corners=False
                    )
                    .sigmoid()
                    .squeeze()
                )

                # Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (
                    pred_resized.max() - pred_resized.min() + 1e-8
                )

                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(
                    g, size=(h_orig, w_orig), mode="nearest"
                ).squeeze()

                # THAY ĐỔI: Ngưỡng u xương chuẩn hóa (0.5 cho cả hai để đảm bảo tính nhị phân khắt khe)
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.5).float()

                d = dice_coefficient(input_binary, target_binary).item()
                io = iou(input_binary, target_binary).item()
                sens, spec, prec, hd = get_binary_metrics(input_binary, target_binary)

                DSC += d
                IOU += io
                total_images += 1

                detailed_results.append(
                    {
                        "Name": names[i],
                        "Dice": d,
                        "IoU": io,
                        "Sensitivity": float("{:.4f}".format(sens)),
                        "Specificity": float("{:.4f}".format(spec)),
                        "Precision": float("{:.4f}".format(prec)),
                        "HD95": float("{:.4f}".format(hd)),
                    }
                )

                if save_base:
                    pred_img = (input_binary.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, names[i]), pred_img)

    return DSC / total_images, IOU / total_images, detailed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, required=True, help="ID folder chứa model đã train"
    )
    parser.add_argument("--network", type=str, default="MK_UNet")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    # THAY ĐỔI: Bone u xương dùng size 256 theo bài báo
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--test_batchsize", type=int, default=1)
    parser.add_argument("--color_image", default=True)
    # THAY ĐỔI: Mặc định trỏ vào thư mục processed của dự án U Xương
    parser.add_argument("--test_path", type=str, default="./data/processed/")
    opt = parser.parse_args()

    # THAY ĐỔI: Cập nhật đường dẫn lưu dự đoán cho U Xương
    save_base = f"./predictions_bone/{opt.run_id}/{opt.split}"
    os.makedirs(save_base, exist_ok=True)
    os.makedirs("results_bone", exist_ok=True)

    # THAY ĐỔI: Tự động tìm file .pth trong folder run_id (hỗ trợ cả -best và -last)
    model_path = os.path.join(f"./model_pth/{opt.run_id}/", f"{opt.run_id}-best.pth")

    NET_CONFIGS = {
        "MK_UNet_T": [4, 8, 16, 24, 32],
        "MK_UNet_S": [8, 16, 32, 48, 80],
        "MK_UNet": [16, 32, 64, 96, 160],
        "MK_UNet_M": [32, 64, 128, 192, 320],
        "MK_UNet_L": [64, 128, 256, 384, 512],
    }

    channels = NET_CONFIGS.get(opt.network, NET_CONFIGS["MK_UNet"])
    # THAY ĐỔI: Sử dụng .to(device) thay vì .cuda() để linh hoạt thiết bị
    model = MK_UNet(num_classes=1, in_channels=3, channels=channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # --- Run Inference ---
    mean_dice, mean_iou, results = test(
        model, opt.test_path, opt.split, opt, save_base=save_base
    )

    # --- Save to Excel ---
    df = pd.DataFrame(results)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row["Name"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # THAY ĐỔI: Tên file kết quả lưu vào kết quả u xương
    excel_name = f"results_bone/Results_{opt.run_id}_{opt.split}.xlsx"
    df.to_excel(excel_name, index=False)

    print(f"\nFinal Results for Bone Tumor ({opt.run_id}):")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Excel report saved to: {excel_name}")

    # --- Persistent Summary ---
    summary_file = "Summary_Bone_Runs.xlsx"
    avg_data = {
        "run_id": opt.run_id,
        "network": opt.network,
        "split": opt.split,
        "dice": mean_dice,
        "iou": mean_iou,
        "sensitivity": mean_row["Sensitivity"],
        "precision": mean_row["Precision"],
        "HD95": mean_row["HD95"],
    }
    df_new = pd.DataFrame([avg_data])
    if os.path.exists(summary_file):
        df_old = pd.read_excel(summary_file)
        pd.concat([df_old, df_new], ignore_index=True).to_excel(
            summary_file, index=False
        )
    else:
        df_new.to_excel(summary_file, index=False)
