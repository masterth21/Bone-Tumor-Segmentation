# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch.nn.functional as F

# # Import từ project của bạn
# from mkunet_network import MK_UNet
# from utils.dataloader_bone import get_loader

# # 1. Cấu hình các đường dẫn
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Thay đổi đường dẫn này tới file .pth tốt nhất trong thư mục model_pth của bạn
# model_path = "./model_pth/BoneTumor_0311_0858/BoneTumor_0311_0858-best.pth" 
# data_path = "./data/processed/test/"
# output_dir = "./results_visualization/"
# os.makedirs(output_dir, exist_ok=True)

# # 2. Tải mô hình
# channels = [16, 32, 64, 96, 160]
# model = MK_UNet(num_classes=1, in_channels=3, channels=channels).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # 3. Chuẩn bị Dataloader cho tập Test
# test_loader = get_loader(
#     image_root=os.path.join(data_path, "images/"),
#     gt_root=os.path.join(data_path, "masks/"),
#     batchsize=1,
#     trainsize=256,
#     shuffle=False,
#     split="test"
# )

# print(f"--- Đang bắt đầu quá trình trực quan hóa kết quả ---")

# # 4. Chạy dự đoán và vẽ ảnh
# with torch.no_grad():
#     for i, pack in enumerate(test_loader):
#         if i >= 10: 
#             break  # Chỉ xuất 10 ảnh mẫu để xem thử

#         # 1. Giải nén dữ liệu từ pack và đưa lên thiết bị (GPU/CPU)
#         images = pack[0].to(device)
#         gts = pack[1].to(device).float()

#         # 2. Dự đoán với mô hình MK-UNet
#         output = model(images)

#         # Xử lý nếu mô hình trả về danh sách kết quả (Deep Supervision)
#         if isinstance(output, (list, tuple)):
#             output = output[0]

#         # Áp dụng Sigmoid và ngưỡng 0.5 để lấy mask nhị phân
#         pred = torch.sigmoid(output)
#         pred = (pred > 0.5).float()

#         # 3. Chuẩn bị dữ liệu Numpy để vẽ đồ thị
#         # Chuyển Tensor (C, H, W) về Numpy (H, W, C)
#         img_tensor = images[0].cpu().permute(1, 2, 0).numpy()
        
#         # Chuẩn hóa ảnh X-ray về dải [0, 1] để hiển thị đúng
#         img_min, img_max = img_tensor.min(), img_tensor.max()
#         if img_max > img_min:
#             img_np = (img_tensor - img_min) / (img_max - img_min)
#         else:
#             img_np = img_tensor

#         gt_np = gts[0].cpu().numpy().squeeze()
#         pred_np = pred[0].cpu().numpy().squeeze()

#         # 4. Vẽ đồ thị so sánh 3 thành phần
#         plt.figure(figsize=(15, 5))

#         # Khung 1: Ảnh X-ray gốc
#         plt.subplot(1, 3, 1)
#         plt.title("Original X-ray")
#         plt.imshow(img_np)
#         plt.axis('off')

#         # Khung 2: Nhãn thực tế (Ground Truth)
#         plt.subplot(1, 3, 2)
#         plt.title("Ground Truth (Label)")
#         plt.imshow(gt_np, cmap='gray')
#         plt.axis('off')

#         # Khung 3: Kết quả dự đoán (Chồng màu lên ảnh gốc)
#         plt.subplot(1, 3, 3)
#         plt.title(f"Prediction (Dice: 0.6190)") 
#         # Hiển thị ảnh gốc làm nền
#         plt.imshow(img_np)
#         # Chồng mask dự đoán màu 'jet' với độ trong suốt 0.5
#         plt.imshow(pred_np, cmap='jet', alpha=0.5) 
#         plt.axis('off')

#         # 5. Lưu ảnh vào thư mục results_visualization
#         output_path = os.path.join(output_dir, f"result_{i}.png")
#         plt.savefig(output_path, bbox_inches='tight', dpi=150)
#         plt.close() # Giải phóng bộ nhớ RAM
        
#         print(f"Đã lưu thành công: {output_path}")

# print(f"--- Hoàn thành! Kiểm tra ảnh tại thư mục: {output_dir} ---")


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Import từ project của bạn
from mkunet_network import MK_UNet
from utils.dataloader_bone import get_loader

# 1. Hàm tính Dice Score chuẩn
def calculate_individual_dice(pred, gt):
    smooth = 1e-6
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = (pred_flat * gt_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    return dice

if __name__ == "__main__":
    # 2. Cấu hình thiết bị và đường dẫn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "./model_pth/BoneTumor_0311_0858/BoneTumor_0311_0858-best.pth" 
    data_path = "./data/processed/test/"
    output_dir = "./results_visualization/"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Khởi tạo mô hình
    channels = [16, 32, 64, 96, 160]
    model = MK_UNet(num_classes=1, in_channels=3, channels=channels).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"--- Đã tải thành công model từ: {model_path} ---")
    else:
        print(f"!!! KHÔNG TÌM THẤY MODEL TẠI: {model_path} !!!")
        exit()
        
    model.eval()

    # 4. Khởi tạo test_loader (ĐÂY LÀ DÒNG ĐỊNH NGHĨA BIẾN test_loader)
    test_loader = get_loader(
        image_root=os.path.join(data_path, "images/"),
        gt_root=os.path.join(data_path, "masks/"),
        batchsize=1,
        trainsize=256,
        shuffle=False,
        split="test"
    )

    print(f"--- Bắt đầu xử lý toàn bộ {len(test_loader)} ảnh trong tập Test ---")

    # 5. Vòng lặp xử lý 300 ảnh
    with torch.no_grad():
        for i, pack in enumerate(test_loader):
            # Lấy dữ liệu từ pack
            images = pack[0].to(device)
            gts = pack[1].to(device).float()

            # Dự đoán
            output = model(images)
            if isinstance(output, (list, tuple)):
                output = output[0]

            pred = torch.sigmoid(output)
            pred_binary = (pred > 0.5).float()

            # Chuyển về Numpy
            img_tensor = images[0].cpu().permute(1, 2, 0).numpy()
            if img_tensor.max() > img_tensor.min():
                img_np = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
            else:
                img_np = img_tensor

            gt_np = gts[0].cpu().numpy().squeeze()
            pred_np = pred_binary[0].cpu().numpy().squeeze()

            # Tính Dice Score chuẩn cho ảnh này
            current_dice = calculate_individual_dice(pred_np, gt_np)

            # 6. Vẽ đồ thị
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title(f"Original X-ray (ID: {i})")
            plt.imshow(img_np)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(gt_np, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title(f"Prediction (Dice: {current_dice:.4f})")
            plt.imshow(img_np)
            plt.imshow(pred_np, cmap='jet', alpha=0.4) 
            plt.axis('off')

            # Lưu ảnh
            save_path = os.path.join(output_dir, f"result_{i:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            if (i + 1) % 20 == 0:
                print(f"Tiến độ: {i+1}/{len(test_loader)} - Dice hiện tại: {current_dice:.4f}")

print(f"--- Hoàn thành! Kiểm tra 300 ảnh tại: {output_dir} ---")