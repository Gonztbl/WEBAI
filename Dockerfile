FROM python:3.10-slim

WORKDIR /app

# 1. Cài đặt các phụ thuộc hệ thống
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy các file nhỏ trước (tối ưu Docker cache layer)
COPY requirements.txt .
COPY app.py .
COPY static/ static/
COPY templates/ templates/

# 3. Tạo thư mục model và tải file model với kiểm tra
RUN mkdir -p model && \
    cd model && \
    wget -q --show-progress --progress=bar:force:noscroll \
    https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/fruit_state_classifier.keras && \
    wget -q --show-progress --progress=bar:force:noscroll \
    https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/yolo11n.pt && \
    wget -q --show-progress --progress=bar:force:noscroll \
    https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/fruit_ripeness_model_pytorch.pth && \
    echo "Verifying files..." && \
    ls -lh && \
    file fruit_state_classifier.keras && \
    [ -s fruit_state_classifier.keras ] || { echo "Model file is empty or missing"; exit 1; }

# 4. Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Thêm healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "300", "--preload", "app:app"]
