# augment_data.py (phiên bản đã sửa lỗi)
import os
import cv2
import numpy as np
import albumentations as A
from glob import glob
from tqdm import tqdm


# ==============================================================================
# --- PHẦN 1: CẤU HÌNH ---
# ==============================================================================

class AugmentConfig:
    # --- Đường dẫn ---
    SOURCE_IMAGE_DIR = 'yolo_dataset/images/train'
    SOURCE_LABEL_DIR = 'yolo_dataset/labels/train'

    AUGMENTED_IMAGE_DIR = 'yolo_dataset_augmented/images/train'
    AUGMENTED_LABEL_DIR = 'yolo_dataset_augmented/labels/train'

    # --- Cấu hình Augmentation ---
    AUGMENTATIONS_PER_IMAGE = 3
    IMAGE_SIZE = 640
    # THÊM CÁC ĐUÔI FILE ẢNH PHỔ BIẾN
    IMAGE_FORMATS = ['*.jpg', '*.jpeg', '*.png']


# ==============================================================================
# --- PHẦN 2: ĐỊNH NGHĨA CHUỖI AUGMENTATION (ĐÃ SỬA LỖI) ---
# ==============================================================================

def get_advanced_transforms(img_size=640):
    """
    Định nghĩa một chuỗi các phép biến đổi ảnh nâng cao bằng Albumentations.
    Đã cập nhật các tham số để loại bỏ warnings.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # SỬA LỖI: Dùng 'cval' thay cho 'value'
        A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT, cval=0),
        # SỬA LỖI: Dùng 'cval' thay cho 'value'
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT,
                           cval=0),

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

        A.CLAHE(p=0.2),
        A.RandomGamma(p=0.2),
        # SỬA LỖI: Dùng 'gauss' thay cho 'var_limit'
        A.GaussNoise(p=0.2),
        A.ISONoise(p=0.2),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.MedianBlur(blur_limit=5, p=0.1),
        A.ToGray(p=0.05),

        A.Resize(height=img_size, width=img_size),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))


# ==============================================================================
# --- PHẦN 3: HÀM CHÍNH ĐỂ TẠO DỮ LIỆU (ĐÃ SỬA LỖI) ---
# ==============================================================================

def create_augmented_dataset(config: AugmentConfig):
    print("--- BẮT ĐẦU TẠO BỘ DỮ LIỆU AUGMENTED ---")

    os.makedirs(config.AUGMENTED_IMAGE_DIR, exist_ok=True)
    os.makedirs(config.AUGMENTED_LABEL_DIR, exist_ok=True)

    transform = get_advanced_transforms(config.IMAGE_SIZE)

    # SỬA LỖI: Tìm kiếm tất cả các định dạng ảnh đã định nghĩa
    image_paths = []
    print(f"Đang tìm kiếm ảnh trong thư mục: {os.path.abspath(config.SOURCE_IMAGE_DIR)}")
    for fmt in config.IMAGE_FORMATS:
        image_paths.extend(glob(os.path.join(config.SOURCE_IMAGE_DIR, fmt)))

    if not image_paths:
        print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy bất kỳ ảnh nào trong '{config.SOURCE_IMAGE_DIR}'.")
        print("Vui lòng kiểm tra lại đường dẫn trong 'AugmentConfig'.")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh gốc. Sẽ tạo {config.AUGMENTATIONS_PER_IMAGE} phiên bản cho mỗi ảnh.")

    for img_path in tqdm(image_paths, desc="Augmenting images"):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_filename = os.path.basename(img_path)
        # SỬA LỖI: Thay thế nhiều định dạng ảnh
        label_filename = os.path.splitext(base_filename)[0] + '.txt'
        label_path = os.path.join(config.SOURCE_LABEL_DIR, label_filename)

        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_center, y_center, w, h = map(float, parts)
                        bboxes.append([x_center, y_center, w, h])
                        class_labels.append(int(cls))

        for i in range(config.AUGMENTATIONS_PER_IMAGE):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']

                new_filename_base = f"{os.path.splitext(base_filename)[0]}_aug_{i}"
                new_img_path = os.path.join(config.AUGMENTED_IMAGE_DIR, new_filename_base + '.jpg')
                new_label_path = os.path.join(config.AUGMENTED_LABEL_DIR, new_filename_base + '.txt')

                cv2.imwrite(new_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                with open(new_label_path, 'w') as f:
                    if 'class_labels' in augmented:
                        for j, bbox in enumerate(aug_bboxes):
                            cls = augmented['class_labels'][j]
                            x_center, y_center, w, h = bbox
                            f.write(f"{cls} {x_center} {y_center} {w} {h}\n")

            except Exception as e:
                print(f"Lỗi khi augment ảnh {img_path} lần {i}: {e}")
                continue

    print("\n--- HOÀN TẤT! ---")
    print(f"Đã tạo xong dữ liệu augmented tại: '{os.path.dirname(config.AUGMENTED_IMAGE_DIR)}'")


if __name__ == '__main__':
    config = AugmentConfig()
    create_augmented_dataset(config)