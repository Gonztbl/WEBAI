# train_yolo.py
from ultralytics import YOLO
import torch
import os


def train_yolo_model():
    """
    Hàm để huấn luyện mô hình YOLOv8 trên bộ dữ liệu trái cây đã chuẩn bị.
    """
    # --- PHẦN CẤU HÌNH HUẤN LUYỆN ---

    # 1. Đường dẫn đến file cấu hình dữ liệu .yaml
    DATA_CONFIG_PATH = 'fruit_data.yaml'

    # 2. Chọn mô hình YOLO để bắt đầu (transfer learning)
    # 'yolov8n.pt': nano, nhỏ và nhanh nhất, tốt cho việc thử nghiệm.
    # 'yolov8s.pt': small, cân bằng giữa tốc độ và độ chính xác.
    # 'yolov8m.pt': medium, chính xác hơn nhưng chậm hơn.
    # Bắt đầu với 'yolov8n.pt' là một lựa chọn tốt.
    BASE_MODEL = 'yolov8n.pt'

    # 3. Các tham số huấn luyện
    EPOCHS = 50  # Số lần lặp qua toàn bộ bộ dữ liệu. Bắt đầu với 50 là hợp lý.
    IMAGE_SIZE = 640  # Kích thước ảnh đầu vào. 640 là tiêu chuẩn cho YOLOv8.
    BATCH_SIZE = 16  # Số lượng ảnh xử lý trong một lần. Nếu gặp lỗi "Out of Memory",
    # hãy giảm con số này xuống (ví dụ: 8, 4).
    PROJECT_NAME = 'fruit_detection_results'  # Tên thư mục để lưu kết quả huấn luyện
    EXPERIMENT_NAME = 'exp_with_augmentation'  # Tên của lần chạy thử nghiệm này

    # --- KẾT THÚC PHẦN CẤU HÌNH ---

    # Kiểm tra file cấu hình dữ liệu có tồn tại không
    if not os.path.exists(DATA_CONFIG_PATH):
        print(f"LỖI: Không tìm thấy file cấu hình '{DATA_CONFIG_PATH}'.")
        print("Vui lòng tạo file này trước khi chạy huấn luyện.")
        return

    # Kiểm tra xem có GPU không
    if torch.cuda.is_available():
        print("Đã phát hiện GPU. Sẽ sử dụng GPU để huấn luyện.")
        device = 0  # Sử dụng GPU đầu tiên, có thể là '0' hoặc [0]
    else:
        print("Không phát hiện GPU. Sẽ sử dụng CPU để huấn luyện (có thể rất chậm).")
        device = 'cpu'

    # Bước 1: Tải mô hình cơ sở
    print(f"Đang tải mô hình cơ sở: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # Bước 2: Bắt đầu quá trình huấn luyện
    print("\n" + "=" * 50)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN YOLOv8")
    print(f"  - File cấu hình dữ liệu: {DATA_CONFIG_PATH}")
    print(f"  - Số Epochs: {EPOCHS}")
    print(f"  - Kích thước ảnh: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print("=" * 50 + "\n")

    try:
        model.train(
            data=DATA_CONFIG_PATH,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=EXPERIMENT_NAME,
            device=device,
            patience=10,  # Dừng sớm nếu không có cải thiện sau 10 epochs
            cache=True  # Lưu cache dữ liệu vào RAM để tăng tốc độ huấn luyện
        )
    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình huấn luyện: {e}")
        print("Gợi ý: Nếu gặp lỗi 'CUDA out of memory', hãy thử giảm giá trị 'BATCH_SIZE' trong file này.")
        return

    print("\n--- HOÀN TẤT HUẤN LUYỆN! ---")
    print(f"Kết quả và mô hình tốt nhất đã được lưu tại thư mục: '{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/'")
    print("Hãy tìm file 'best.pt' trong thư mục đó và copy nó vào thư mục 'model' của bạn để sử dụng trong ứng dụng.")


if __name__ == '__main__':
    train_yolo_model()