# Bước 1: Sử dụng một ảnh Docker Python chính thức làm nền
FROM python:3.10-slim

# Bước 2: Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Bước 3: Cập nhật và cài đặt git và git-lfs 
# (Đây là những gì chúng ta thực sự cần)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Bước 4: Sao chép TOÀN BỘ mã nguồn của bạn vào trong container
# Thay vì sao chép từng phần, chúng ta sao chép tất cả ngay bây giờ.
COPY . .

# Bước 5: Khởi tạo Git LFS và tải về các tệp lớn
# Lệnh 'git lfs install' sẽ tạo tệp .gitattributes nếu nó chưa tồn tại.
# Lệnh 'git lfs pull' sẽ tải các mô hình của bạn về.
RUN git lfs install && \
    git lfs pull

# Bước 6: Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Bước 7: Mở cổng 10000 (cổng tiêu chuẩn của Render)
EXPOSE 10000

# Bước 8: Định nghĩa lệnh để khởi chạy ứng dụng của bạn
# Render sẽ sử dụng lệnh này để chạy ứng dụng của bạn.
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "4", "app:app"]
