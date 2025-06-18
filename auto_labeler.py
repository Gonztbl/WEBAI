# auto_labeler.py
import os
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm

# --- PHẦN CẤU HÌNH ---

# 1. Đường dẫn đến bộ dữ liệu gốc của bạn
# Script sẽ tìm trong các thư mục con của đường dẫn này
BASE_DATASET_PATH = 'dataset/train'

# 2. Thư mục để lưu các file nhãn .txt được tạo ra
OUTPUT_DIR = 'autolabels'

# 3. Danh sách các lớp của bạn. THỨ TỰ NÀY RẤT QUAN TRỌNG!
# Nó quyết định chỉ số lớp (class index) sẽ được ghi vào file .txt
MY_LABEL_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_orange',
    'rotten_apple', 'rotten_banana', 'rotten_orange'
]

# 4. Các loại file ảnh cần xử lý
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']


# --- KẾT THÚC PHẦN CẤU HÌNH ---


def run_auto_labeling():
    """
    Chạy qua bộ dữ liệu, dùng YOLOv8 để dự đoán bounding box
    và tạo file nhãn tự động.
    """
    # Tải mô hình YOLOv8 đã được huấn luyện sẵn trên COCO
    # Nó sẽ tự động tải về nếu bạn chưa có
    print("Đang tải model YOLOv8n...")
    model = YOLO('yolov8n.pt')

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Các file nhãn sẽ được lưu vào thư mục: '{OUTPUT_DIR}'")

    # Lấy danh sách các thư mục con trong dataset (freshapples, rottenoranges, ...)
    subdirs = [d for d in os.listdir(BASE_DATASET_PATH) if os.path.isdir(os.path.join(BASE_DATASET_PATH, d))]

    # Bắt đầu xử lý từng thư mục
    for subdir_name in subdirs:
        # Xác định loại quả và tình trạng từ tên thư mục
        fruit_type, state = None, None

        # Xác định loại quả
        if 'apple' in subdir_name.lower():
            fruit_type = 'apple'
        elif 'banana' in subdir_name.lower():
            fruit_type = 'banana'
        elif 'orange' in subdir_name.lower():
            fruit_type = 'orange'

        # Xác định tình trạng
        if 'fresh' in subdir_name.lower():
            state = 'fresh'
        elif 'rotten' in subdir_name.lower():
            state = 'rotten'

        # Nếu không xác định được, bỏ qua thư mục này
        if not fruit_type or not state:
            print(f"Bỏ qua thư mục không xác định: {subdir_name}")
            continue

        # Tạo nhãn mục tiêu và tìm chỉ số lớp của nó
        target_label_name = f"{state}_{fruit_type}"
        try:
            target_class_index = MY_LABEL_NAMES.index(target_label_name)
        except ValueError:
            print(f"Lỗi: Nhãn '{target_label_name}' từ thư mục '{subdir_name}' không có trong MY_LABEL_NAMES.")
            continue

        # Lấy danh sách tất cả các file ảnh trong thư mục con
        image_files = []
        full_subdir_path = os.path.join(BASE_DATASET_PATH, subdir_name)
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob(os.path.join(full_subdir_path, ext)))

        print(
            f"\nĐang xử lý {len(image_files)} ảnh trong thư mục '{subdir_name}' -> Gán nhãn là '{target_label_name}' (index: {target_class_index})")

        # Sử dụng tqdm để tạo thanh tiến trình
        for img_path in tqdm(image_files, desc=f"Processing {subdir_name}"):
            # Chạy dự đoán trên ảnh
            results = model(img_path, verbose=False)  # verbose=False để không in quá nhiều log

            # Tạo đường dẫn cho file .txt đầu ra
            base_filename = os.path.basename(img_path)
            txt_filename = os.path.splitext(base_filename)[0] + '.txt'
            output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)

            with open(output_txt_path, 'w') as f:
                for r in results:
                    for box in r.boxes:
                        # Lấy tên lớp mà model COCO nhận ra
                        coco_class_name = model.names[int(box.cls[0])]

                        # Chỉ ghi lại nếu model COCO nhận ra đúng loại quả
                        if coco_class_name == fruit_type:
                            # Lấy tọa độ đã được chuẩn hóa [x_center, y_center, width, height]
                            xywhn = box.xywhn[0]
                            x_center, y_center, w, h = xywhn

                            # Ghi vào file với CHỈ SỐ LỚP CỦA BẠN
                            f.write(f"{target_class_index} {x_center} {y_center} {w} {h}\n")

    print("\n\n--- HOÀN TẤT! ---")
    print("Quá trình tự động gán nhãn đã xong.")
    print("Bước tiếp theo: Dùng MakeSense.ai để 'Import Annotations' và kiểm tra lại kết quả.")


if __name__ == '__main__':
    run_auto_labeling()