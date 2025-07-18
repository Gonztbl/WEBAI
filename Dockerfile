# Bước 1: Sử dụng một ảnh Docker Python chính thức làm nền
FROM python:3.10-slim

# Bước 2: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 3: Cập nhật và cài đặt các thư viện hệ thống cần thiết cho OpenCV
# ĐÃ THÊM libgl1-mesa-glx và libglib2.0-0
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Bước 4: Sao chép mã nguồn và các tệp nhỏ vào trước
COPY . .

# Bước 5: Tạo thư mục model
RUN mkdir -p model

# Bước 6: Tải trực tiếp các tệp mô hình lớn từ GitHub Releases
RUN wget -O model/fruit_state_classifier.keras "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/fruit_state_classifier.keras"
RUN wget -O model/yolo11n.pt "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/yolo11n.pt"
RUN wget -O model/fruit_ripeness_model_pytorch.pth "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/fruit_ripeness_model_pytorch.pth"

# Bước 7: Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Bước 8: Mở cổng 10000
EXPOSE 10000

# Bước 9: Định nghĩa lệnh để khởi chạy ứng dụng (Đã tối ưu hóa)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "300", "--preload", "app:app"]
