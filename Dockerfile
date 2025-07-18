# Bước 1: Sử dụng một ảnh Docker Python chính thức làm nền
FROM python:3.10-slim

# Bước 2: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 3: Cập nhật và cài đặt wget (công cụ tải file)
# Chúng ta không cần git hay git-lfs nữa.
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Bước 4: Sao chép mã nguồn và các tệp nhỏ vào trước
COPY . .

# Bước 5: Tạo thư mục model (nếu nó chưa được sao chép)
RUN mkdir -p model

# Bước 6: Tải trực tiếp các tệp mô hình lớn bằng wget
# QUAN TRỌNG: Thay thế các URL placeholder bên dưới bằng các link bạn đã sao chép!
RUN wget -O model/fruit_state_classifier.keras "https://github.com/Gonztbl/WEBAI/blob/72ca00452176cc4d551b21c1e77e811f0ea2d80c/model/fruit_state_classifier.keras"
RUN wget -O model/yolo11n.pt "https://github.com/Gonztbl/WEBAI/blob/72ca00452176cc4d551b21c1e77e811f0ea2d80c/model/yolo11n.pt"
RUN wget -O model/fruit_ripeness_model_pytorch.pth "https://github.com/Gonztbl/WEBAI/blob/72ca00452176cc4d551b21c1e77e811f0ea2d80c/model/fruit_ripeness_model_pytorch.pth"

# Bước 7: Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Bước 8: Mở cổng 10000
EXPOSE 10000

# Bước 9: Định nghĩa lệnh để khởi chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "4", "app:app"]
