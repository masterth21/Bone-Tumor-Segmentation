# import os
# import json
# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# # --- CẤU HÌNH ĐƯỜNG DẪN ---
# # Tự động lấy đường dẫn tuyệt đối của thư mục chứa file script này
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Đường dẫn đến dữ liệu thô (Raw) và dữ liệu sau xử lý (Processed)
# RAW_IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
# RAW_ANN_DIR = os.path.join(BASE_DIR, "data", "raw", "Annotations")
# PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# # Kích thước ảnh 256x256 giúp đạt hiệu suất cao (138.89 images/s) [cite: 270, 284]
# IMG_SIZE = 256


# def create_mask_from_json(json_path, img_shape):
#     """Chuyển đổi file JSON LabelMe thành ảnh Mask nhị phân"""
#     mask = np.zeros(img_shape[:2], dtype=np.uint8)
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         for shape in data.get("shapes", []):
#             points = np.array(shape["points"], dtype=np.int32)

#             if shape.get("shape_type") == "rectangle":
#                 # Đối với rectangle, points[0] là góc trên trái, points[1] là góc dưới phải
#                 cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), 255, -1)
#             else:
#                 # Đối với polygon (đa giác), vẽ vùng kín
#                 cv2.fillPoly(mask, [points], 255)
#     except Exception as e:
#         print(f"Error processing {json_path}: {e}")
#     return mask


# def setup_folders():
#     """Tạo cấu trúc thư mục processed/train, val, test theo đúng chuẩn 8:1:1 [cite: 263]"""
#     for split in ["train", "val", "test"]:
#         os.makedirs(os.path.join(PROCESSED_DIR, split, "images"), exist_ok=True)
#         os.makedirs(os.path.join(PROCESSED_DIR, split, "masks"), exist_ok=True)


# def process_data():
#     setup_folders()

#     # Lấy danh sách ID ảnh dựa trên các file có đuôi phổ biến
#     image_files = [
#         f
#         for f in os.listdir(RAW_IMG_DIR)
#         if f.lower().endswith((".jpeg", ".jpg", ".png"))
#     ]
#     image_ids = [os.path.splitext(f)[0] for f in image_files]

#     # Chia tập dữ liệu: 80% Train, 10% Val, 10% Test theo đúng bài báo [cite: 263]
#     train_ids, temp_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
#     val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

#     split_dict = {"train": train_ids, "val": val_ids, "test": test_ids}

#     for split, ids in split_dict.items():
#         print(f"Processing split: {split}...")
#         for img_id in tqdm(ids):
#             # 1. Đọc ảnh gốc với các biến thể đuôi file để tránh lỗi imread
#             img = None
#             found_ext = None
#             for ext in [".jpeg", ".jpg", ".JPEG", ".JPG"]:
#                 path_to_check = os.path.join(RAW_IMG_DIR, f"{img_id}{ext}")
#                 if os.path.exists(path_to_check):
#                     img = cv2.imread(path_to_check)
#                     if img is not None:
#                         found_ext = ext
#                         break

#             # Nếu không đọc được ảnh, in cảnh báo và bỏ qua ID này
#             if img is None:
#                 print(f" Warning: Could not read image {img_id}. Skipping...")
#                 continue

#             # Resize ảnh về kích thước chuẩn 256x256 theo nghiên cứu
#             h, w = img.shape[:2]
#             img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

#             # 2. Tạo Mask từ file JSON tương ứng
#             json_path = os.path.join(RAW_ANN_DIR, f"{img_id}.json")
#             mask = create_mask_from_json(json_path, (h, w))

#             # Sử dụng INTER_NEAREST để giữ giá trị mask nhị phân (0 hoặc 255)
#             mask_resized = cv2.resize(
#                 mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
#             )

#             # 3. Lưu ảnh và mask vào thư mục processed dưới dạng .png
#             # Định dạng .png giúp bảo toàn pixel cho các vùng biên khối u [cite: 313]
#             cv2.imwrite(
#                 os.path.join(PROCESSED_DIR, split, "images", f"{img_id}.png"),
#                 img_resized,
#             )
#             cv2.imwrite(
#                 os.path.join(PROCESSED_DIR, split, "masks", f"{img_id}.png"),
#                 mask_resized,
#             )


# if __name__ == "__main__":
#     process_data()
#     print("Data preprocessing completed successfully!")


import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_ANN_DIR = os.path.join(BASE_DIR, "data", "raw", "Annotations")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Kích thước chuẩn 256x256 giúp đạt hiệu suất cao theo bài báo [cite: 270]
IMG_SIZE = 256


def create_mask_from_json(json_path, img_shape):
    """Chuyển đổi JSON sang Mask - Ưu tiên Polygon để phân đoạn chuẩn"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Lọc danh sách các hình vẽ
        shapes = data.get("shapes", [])

        # Cách 1: Chỉ vẽ Polygon (Khuyên dùng cho segmentation chuẩn)
        for shape in shapes:
            if shape.get("shape_type") == "polygon":
                points = np.array(shape["points"], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

        # Nếu sau khi quét xong mà mask vẫn đen thui (không có polygon),
        # lúc đó mới vẽ rectangle làm phương án dự phòng (optional)
        if np.sum(mask) == 0:
            for shape in shapes:
                if shape.get("shape_type") == "rectangle":
                    points = np.array(shape["points"], dtype=np.int32)
                    cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), 255, -1)

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
    return mask


def process_data():
    # Tạo cấu trúc folder
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(PROCESSED_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, split, "masks"), exist_ok=True)

    # Chỉ lấy các ID từ 1 đến 1867 có đầy đủ cặp ảnh-nhãn
    valid_ids = []
    for i in range(1, 1868):
        img_id = f"IMG{i:06d}"
        # Kiểm tra sự tồn tại của cả ảnh và nhãn
        json_path = os.path.join(RAW_ANN_DIR, f"{img_id}.json")
        has_img = any(
            [
                os.path.exists(os.path.join(RAW_IMG_DIR, f"{img_id}{ext}"))
                for ext in [".jpeg", ".jpg", ".JPEG", ".JPG"]
            ]
        )

        if has_img and os.path.exists(json_path):
            valid_ids.append(img_id)

    print(f"Found {len(valid_ids)} valid pairs in range 1-1867.")

    # Chia tập 80:10:10 (Train: 1493, Val: 187, Test: 187)
    train_ids, temp_ids = train_test_split(valid_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    split_dict = {"train": train_ids, "val": val_ids, "test": test_ids}

    for split, ids in split_dict.items():
        print(f"Processing {split} split...")
        for img_id in tqdm(ids):
            img = None
            for ext in [".jpeg", ".jpg", ".JPEG", ".JPG"]:
                path = os.path.join(RAW_IMG_DIR, f"{img_id}{ext}")
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        break

            if img is None:
                continue

            h, w = img.shape[:2]
            img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            mask = create_mask_from_json(
                os.path.join(RAW_ANN_DIR, f"{img_id}.json"), (h, w)
            )
            if mask is None:
                continue
            mask_res = cv2.resize(
                mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
            )

            cv2.imwrite(
                os.path.join(PROCESSED_DIR, split, "images", f"{img_id}.png"), img_res
            )
            cv2.imwrite(
                os.path.join(PROCESSED_DIR, split, "masks", f"{img_id}.png"), mask_res
            )


if __name__ == "__main__":
    process_data()
    print("Preprocessing finished successfully!")
