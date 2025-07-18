# Bước 1: Sử dụng một ảnh Docker Python chính thức làm nền
FROM python:3.10-slim

# Bước 2: Thiết lập biến môi trường cho URL kho chứa của bạn
# GIẢI PHÁP: Truyền URL kho chứa vào làm một biến để dễ sử dụng
ARG REPO_URL="https://github.com/Gonztbl/WEBAI.git"

# Bước 3: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 4: Cập nhật và cài đặt các công cụ cần thiết
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Bước 5: Đây là phần quan trọng nhất để sửa lỗi "Not in a Git repository"
# Chúng ta sẽ khởi tạo một kho chứa Git rỗng, kết nối nó với GitHub,
# sau đó mới tải các tệp LFS.
RUN git init && \
    git remote add origin ${REPO_URL} && \
    git lfs install && \
    # Chúng ta chỉ tải về các tệp LFS, không cần toàn bộ lịch sử Git
    git lfs pull

# Bước 6: Bây giờ, sao chép toàn bộ mã nguồn của bạn vào
# Điều này sẽ ghi đè lên các tệp con trỏ bằng các tệp mô hình thật
COPY . .

# Bước 7: Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Bước 8: Mở cổng 10000
EXPOSE 10000

# Bước 9: Định nghĩa lệnh để khởi chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "4", "app:app"]
