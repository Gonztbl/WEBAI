# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_URL="https://github.com/Gonztbl/WEBAI/releases/download/v.1.1"

# Create and set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download TensorFlow models
RUN python -c "try: from tensorflow.keras.applications import MobileNetV2; MobileNetV2(weights='imagenet'); print('TensorFlow models downloaded')\nexcept Exception as e: print(f'TensorFlow download failed: {e}')" || true

# Download model files with better error handling
RUN mkdir -p model && \
    cd model && \
    echo "Downloading models..." && \
    (wget -q "${MODEL_URL}/fruit_state_classifier.weights.h5" && echo "Classifier downloaded" || echo "Classifier download failed") && \
    (wget -q "${MODEL_URL}/yolov8l.pt" && echo "YOLO downloaded" || echo "YOLO download failed") && \
    (wget -q "${MODEL_URL}/fruit_ripeness_model_pytorch.pth" && echo "PyTorch downloaded" || echo "PyTorch download failed") && \
    echo "Download process completed" && \
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
    CMD curl -f http://localhost:10000/health || exit 1

# Start command with optimized settings
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "300", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
