FROM python:3.10-slim

WORKDIR /app

# 1. Cài đặt các phụ thuộc hệ thống
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements.txt trước để tận dụng Docker cache
COPY requirements.txt .

# 3. Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy source code
COPY . .

# 5. Tải model files với kiểm tra nghiêm ngặt
RUN mkdir -p model && \
    cd model && \
    for model in fruit_state_classifier.keras yolo11n.pt fruit_ripeness_model_pytorch.pth; do \
        echo "Downloading ${model}..."; \
        wget -q --show-progress --progress=bar:force:noscroll \
        "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/${model}" && \
        [ -s "${model}" ] || { echo "ERROR: Model file ${model} is empty or missing"; exit 1; } \
    done && \
    echo "Model files verification:" && \
    ls -lh && \
    file fruit_state_classifier.keras yolo11n.pt

# 6. Tạo thư mục lưu ảnh
RUN mkdir -p static/images

# 7. Healthcheck và port
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

EXPOSE 10000

# 8. Lệnh khởi chạy với preload và timeout
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "300", "--preload", "app:app"]
