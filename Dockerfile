# Sử dụng Python 3.10 slim
FROM python:3.10-slim

# Đặt các biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_URL="https://github.com/Gonztbl/WEBAI/releases/download/v.1.1"
ENV TF_CPP_MIN_LOG_LEVEL=3

# ===> SỬA LỖI: Cài đặt BỘ ĐẦY ĐỦ các thư viện hệ thống cho Headless OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    # Các thư viện cần thiết cho OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt và cài đặt các gói Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép phần còn lại của ứng dụng
COPY . .

# Tải các mô hình
RUN mkdir -p model && \
    cd model && \
    echo "Downloading model files..." && \
    wget -q "${MODEL_URL}/fruit_state_classifier_new.h5" && \
    wget -q "${MODEL_URL}/fruit_class_indices.json" && \
    wget -q "${MODEL_URL}/fruit_ripeness_model_pytorch.pth" && \
    wget -q "${MODEL_URL}/yolo11n.pt" && \
    echo "Download completed."

# Tạo thư mục và thiết lập quyền
RUN mkdir -p static/images && \
    chmod -R 755 static && \
    useradd -m appuser && \
    chown -R appuser:appuser /app

# Chuyển sang người dùng không phải root
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1


CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--worker-class", "gevent", \
     "--timeout", "120", \
     "app:app"]```



