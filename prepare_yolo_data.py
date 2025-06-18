# prepare_yolo_data.py
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

# --- PHẦN CẤU HÌNH ---

# 1. Đường dẫn đến bộ dữ liệu gốc (nơi chứa các thư mục lớp)
SOURCE_TRAIN_DATA_DIR = 'dataset/train'
# SOURCE_TEST_DATA_DIR = 'dataset/test' # Chúng ta sẽ dùng tập test này sau

# 2. Đường dẫn đến thư mục chứa các nhãn đã được tạo tự động
SOURCE_LABELS_DIR = 'autolabels'

# 3. Thư mục đầu ra cho bộ dữ liệu YOLO
YOLO_DATASET_DIR = 'yolo_dataset'

# 4. Tỷ lệ dữ liệu dành cho tập validation (ví dụ: 0.2 nghĩa là 20%)
VALIDATION_SPLIT_RATIO = 0.2

# 5. Các loại file ảnh cần tìm
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']


# --- KẾT THÚC PHẦN CẤU HÌNH ---


def create_yolo_structure():
    """Tạo cấu trúc thư mục cần thiết cho YOLO."""
    # Tạo thư mục gốc yolo_dataset
    os.makedirs(YOLO_DATASET_DIR, exist_ok=True)

    # Tạo các thư mục con
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', subset), exist_ok=True)
    print(f"Đã tạo cấu trúc thư mục trong '{YOLO_DATASET_DIR}'")


def process_and_split_data():
    """
    Duyệt qua dữ liệu gốc, chia train/val và copy ảnh cùng nhãn
    vào đúng cấu trúc thư mục của YOLO.
    """
    create_yolo_structure()

    # Lấy danh sách các thư mục lớp (freshapples, rottenoranges, ...)
    class_dirs = [d for d in os.listdir(SOURCE_TRAIN_DATA_DIR) if os.path.isdir(os.path.join(SOURCE_TRAIN_DATA_DIR, d))]

    print(f"\nBắt đầu xử lý và chia {len(class_dirs)} lớp dữ liệu...")

    for class_dir in class_dirs:
        full_class_path = os.path.join(SOURCE_TRAIN_DATA_DIR, class_dir)

        # Lấy danh sách tất cả các file ảnh trong thư mục lớp
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob(os.path.join(full_class_path, ext)))

        if not image_files:
            print(f"Không tìm thấy ảnh nào trong: {class_dir}. Bỏ qua.")
            continue

        # Chia danh sách ảnh thành tập train và validation
        train_files, val_files = train_test_split(
            image_files,
            test_size=VALIDATION_SPLIT_RATIO,
            random_state=42  # Đặt random_state để kết quả chia luôn giống nhau
        )

        print(f"\nLớp '{class_dir}': {len(train_files)} ảnh train, {len(val_files)} ảnh val.")

        # Copy file vào các thư mục tương ứng
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')

    print("\n--- HOÀN TẤT! ---")
    print("Đã sắp xếp xong dữ liệu vào thư mục 'yolo_dataset'.")
    print("Bước tiếp theo: Tạo file 'fruit_data.yaml' và chạy 'train_yolo.py'.")


def copy_files(file_list, subset):
    """
    Copy một danh sách file ảnh và file nhãn tương ứng vào thư mục đích.
    subset: chuỗi 'train' hoặc 'val'.
    """
    for img_path in tqdm(file_list, desc=f"Copying to {subset}"):
        # --- Copy file ảnh ---
        dest_img_dir = os.path.join(YOLO_DATASET_DIR, 'images', subset)
        shutil.copy(img_path, dest_img_dir)

        # --- Copy file nhãn ---
        base_filename = os.path.basename(img_path)
        txt_filename = os.path.splitext(base_filename)[0] + '.txt'
        source_label_path = os.path.join(SOURCE_LABELS_DIR, txt_filename)

        dest_label_dir = os.path.join(YOLO_DATASET_DIR, 'labels', subset)

        if os.path.exists(source_label_path):
            shutil.copy(source_label_path, dest_label_dir)
        else:
            # Tạo file nhãn rỗng nếu không tìm thấy (trường hợp AI không nhận ra đối tượng)
            # Điều này quan trọng để YOLO không báo lỗi thiếu file.
            open(os.path.join(dest_label_dir, txt_filename), 'w').close()


if __name__ == '__main__':
    # Kiểm tra xem thư mục autolabels có tồn tại không
    if not os.path.isdir(SOURCE_LABELS_DIR):
        print(f"Lỗi: Không tìm thấy thư mục nhãn '{SOURCE_LABELS_DIR}'.")
        print("Vui lòng chạy 'auto_labeler.py' trước.")
    else:
        process_and_split_data()