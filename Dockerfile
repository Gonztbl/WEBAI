# Sử dụng Python 3.10 slim làm nền tảng, nhẹ và hiệu quả
FROM python:3.10-slim

# Đặt các biến môi trường để tối ưu hóa Python trong Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV MODEL_URL="https://github.com/Gonztbl/WEBAI/releases/download/v.1.1"

# ----------------- LỚP PHỤ THUỘC HỆ THỐNG (CACHE TỐT NHẤT) -----------------
# Cài đặt các gói cần thiết cho hệ điều hành. Lớp này gần như không bao giờ thay đổi.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && \
    rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc trong container
WORKDIR /app

# ----------------- LỚP PHỤ THUỘC PYTHON (CACHE TỐT) -----------------
# Sao chép CHỈ file requirements.txt
COPY requirements.txt .
# Cài đặt các thư viện Python. Lớp này chỉ chạy lại khi requirements.txt thay đổi.
RUN pip install --no-cache-dir -r requirements.txt

# ----------------- LỚP MÔ HÌNH TĨNH (CACHE TỐT) -----------------
# Tải các mô hình TRƯỚC KHI sao chép mã nguồn ứng dụng.
# Lớp này chỉ chạy lại khi một trong các lớp ở trên thay đổi.
RUN mkdir -p model && \
    cd model && \
    echo "Downloading model files..." && \
    wget -q "${MODEL_URL}/fruit_state_classifier.keras" && \
    wget -q "${MODEL_URL}/fruit_class_indices.json" && \
    wget -q "${MODEL_URL}/fruit_ripeness_model_pytorch.pth" && \
    wget -q "${MODEL_URL}/yolo11n.pt" && \
    echo "Download completed."

# ----------------- LỚP MÃ NGUỒN (THAY ĐỔI THƯỜNG XUYÊN NHẤT) -----------------
# Bây giờ mới sao chép toàn bộ mã nguồn ứng dụng (app.py, v.v.).
# Đặt lớp này ở gần cuối để tận dụng cache tối đa cho các lớp nặng ở trên.
COPY . .

# ----------------- CẤU HÌNH CUỐI CÙNG -----------------
# Tạo các thư mục và thiết lập quyền hạn cần thiết.
RUN mkdir -p static/images && \
    chmod -R 755 static && \
    useradd -m appuser && \
    chown -R appuser:appuser /app

# Chuyển sang người dùng không phải root để tăng cường bảo mật
USER appuser

# Expose port mà Gunicorn sẽ lắng nghe
EXPOSE 5000

# Healthcheck để Render biết khi nào ứng dụng sẵn sàng, đặc biệt quan trọng với
# ứng dụng khởi động chậm. Nó cho phép ứng dụng có 180s để khởi động.
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Lệnh để chạy ứng dụng, đã được tối ưu cho Render Free Tier
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--worker-class", "gevent", \
     "--timeout", "300", \
     "app:app"]
