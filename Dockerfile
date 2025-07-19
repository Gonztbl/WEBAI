# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables for memory optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_URL="https://github.com/Gonztbl/WEBAI/releases/download/v.1.1"
ENV TF_CPP_MIN_LOG_LEVEL=3

# Create working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*
# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY . .

# Download model files - PRIORITIZE .H5 FORMAT
RUN mkdir -p model && \
    cd model && \
    echo "Downloading model files (prioritizing .h5 format)..." && \
    echo "1. Downloading H5 model (.h5 format)..." && \
    (wget -q "${MODEL_URL}/fruit_state_classifier_new.h5" && echo "✅ H5 model downloaded" || echo "❌ H5 model download failed") && \
    echo "2. Downloading class indices (.json)..." && \
    (wget -q "${MODEL_URL}/fruit_class_indices.json" && echo "✅ Class indices downloaded" || echo "❌ Class indices download failed") && \
    echo "3. Downloading PyTorch model (.pth)..." && \
    (wget -q "${MODEL_URL}/fruit_ripeness_model_pytorch.pth" && echo "✅ PyTorch model downloaded" || echo "❌ PyTorch model download failed") && \
    echo "4. Downloading small YOLO model (.pt)..." && \
    (wget -q "${MODEL_URL}/yolo11n.pt" && echo "✅ Small YOLO downloaded" || echo "❌ Small YOLO download failed") && \
    echo "Download completed. Files in model directory:" && \
    ls -la

# Create directories and set permissions
RUN mkdir -p static/images && \
    chmod -R 755 static && \
    useradd -m appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# FIXED: Use existing app.py instead of app_fixed_h5.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "1", \
     "--timeout", "300", \
     "--max-requests", "100", \
     "--max-requests-jitter", "10", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
