import os
import shutil
import yaml

# ==============================================================================
# PHẦN THIẾT LẬP - BẠN CHỈ CẦN THAY ĐỔI Ở ĐÂY
# ==============================================================================

# 1. ĐƯỜNG DẪN ĐẦY ĐỦ ĐẾN THƯ MỤC YOLO BẠN VỪA GIẢI NÉN
# Ví dụ trên Windows: 'C:/Users/YourUser/Downloads/Fruit-Ripeness-O-1'
# Ví dụ trên Mac/Linux: '/home/youruser/downloads/Fruit-Ripeness-O-1'
yolo_dataset_path = 'Fruit Ripeness.v2-compiled-datasets.yolov7pytorch'  # <<<< ----- THAY ĐỔI ĐƯỜNG DẪN NÀY CHO ĐÚNG

# 2. TÊN THƯ MỤC MỚI SẼ ĐƯỢC TẠO RA (bạn có thể giữ nguyên tên này)
classification_dataset_path = 'Fruit_Classification_Data'


# ==============================================================================
# PHẦN XỬ LÝ - Không cần chỉnh sửa gì ở dưới đây
# ==============================================================================

def convert_data():
    """Hàm chính để thực hiện chuyển đổi dữ liệu."""
    print("Bắt đầu quá trình chuyển đổi dữ liệu từ định dạng YOLO sang Phân loại...")

    # Kiểm tra xem đường dẫn đầu vào có tồn tại không
    if not os.path.exists(yolo_dataset_path):
        print(f"LỖI: Đường dẫn '{yolo_dataset_path}' không tồn tại. Vui lòng kiểm tra lại.")
        return

    # Đọc file data.yaml để lấy tên các lớp
    yaml_path = os.path.join(yolo_dataset_path, 'data.yaml')
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
        print(f"Đã tìm thấy các lớp: {class_names}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file 'data.yaml' trong '{yolo_dataset_path}'.")
        return

    # Tạo thư mục gốc cho bộ dữ liệu mới
    if not os.path.exists(classification_dataset_path):
        os.makedirs(classification_dataset_path)
        print(f"Đã tạo thư mục đích: '{classification_dataset_path}'")

    # Lặp qua các tập (train, valid, test)
    for split in ['train', 'valid', 'test']:
        yolo_split_path = os.path.join(yolo_dataset_path, split)
        if not os.path.exists(yolo_split_path):
            continue

        class_split_path = os.path.join(classification_dataset_path, split)
        if not os.path.exists(class_split_path):
            os.makedirs(class_split_path)

        print(f"\nĐang xử lý tập '{split}'...")

        # Tạo các thư mục con cho từng lớp
        for name in class_names:
            os.makedirs(os.path.join(class_split_path, name), exist_ok=True)

        images_path = os.path.join(yolo_split_path, 'images')
        labels_path = os.path.join(yolo_split_path, 'labels')

        if not os.path.isdir(labels_path):
            print(f"Cảnh báo: Không tìm thấy thư mục 'labels' cho tập '{split}'. Bỏ qua.")
            continue

        count = 0
        # Duyệt qua các file chú thích để xác định lớp của ảnh
        for label_file in os.listdir(labels_path):
            if not label_file.endswith('.txt'):
                continue

            # Đọc dòng đầu tiên của file label để lấy lớp
            with open(os.path.join(labels_path, label_file), 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    continue

                class_index = int(first_line.split()[0])
                class_name = class_names[class_index]

                # Tìm ảnh gốc
                image_name_base = os.path.splitext(label_file)[0]
                source_image_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_path = os.path.join(images_path, image_name_base + ext)
                    if os.path.exists(potential_path):
                        source_image_path = potential_path
                        break

                if source_image_path:
                    # Sao chép ảnh vào đúng thư mục lớp
                    destination_path = os.path.join(class_split_path, class_name, os.path.basename(source_image_path))
                    shutil.copy(source_image_path, destination_path)
                    count += 1

        print(f"-> Hoàn thành! Đã sao chép {count} ảnh.")

    print("\n==================================================")
    print("QUÁ TRÌNH CHUYỂN ĐỔI HOÀN TẤT!")
    print(f"Dữ liệu phân loại của bạn đã sẵn sàng tại: '{classification_dataset_path}'")
    print("==================================================")


if __name__ == '__main__':
    convert_data()