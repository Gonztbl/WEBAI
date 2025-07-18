# Bước 1: Sử dụng một ảnh Docker Python chính thức làm nền
FROM python:3.10-slim

# Bước 2: Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Bước 3: Cập nhật và cài đặt git và git-lfs (lệnh này sẽ hoạt động trong Docker!)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    # Dọn dẹp để giữ kích thước ảnh nhỏ
    rm -rf /var/lib/apt/lists/*

# Bước 4: Sao chép các tệp cấu hình git để tối ưu hóa cache của Docker
COPY .git .git
COPY .gitattributes .gitattributes

# Bước 5: Lệnh quan trọng - Kéo các tệp lớn từ Git LFS
RUN git lfs pull

# Bước 6: Sao chép toàn bộ mã nguồn ứng dụng của bạn vào
COPY . .

# Bước 7: Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Bước 8: Mở cổng 10000 (cổng tiêu chuẩn của Render)
EXPOSE 10000

# Bước 9: Định nghĩa lệnh để khởi chạy ứng dụng của bạn
# Lệnh này sẽ thay thế cho "Start Command" trên Render
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "4", "app:app"]