# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MODEL_URL="https://github.com/Gonztbl/WEBAI/releases/download/v.1.1"

# Create and set working directory
WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# 4. Download and VERIFY model files (Final Version with Weights)
RUN mkdir -p model && \
    cd model && \
    \
    echo "Downloading all models..." && \
    wget -q "${MODEL_URL}/fruit_state_classifier.weights.h5" && \
    wget -q "${MODEL_URL}/yolov8l.pt" && \
    wget -q "${MODEL_URL}/fruit_ripeness_model_pytorch.pth" && \
    \
    echo "Verifying all checksums..." && \
    # Create the checksum file with correct hashes from local machine
    echo "54fe888391e8ff7a70f72134e7b2361b6f7b67e76c472b2af71cd2ec1bf76c8b  fruit_state_classifier.weights.h5" > checksums.txt && \
    echo "18218ea4798da042d9862e6029ca9531adbd40ace19b6c9a75e2e28f1adf30cc  yolov8l.pt" >> checksums.txt && \
    echo "48bf9333f4f07af2d02e3965f797fa56fa429d46b34d29d24e95dc925582e63d  fruit_ripeness_model_pytorch.pth" >> checksums.txt && \
    \
    # Verify against the newly created file
    sha256sum -c --strict checksums.txt

# 5. Create directories for static files
RUN mkdir -p static/images && \
    chmod -R a+rwx static

# 6. Health check configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# 7. Expose port
EXPOSE 10000

# 8. Run as non-root user for security
RUN useradd -m appuser && \
    chown -R appuser /app && \
    chmod -R a+rwx /app/model
USER appuser

# 9. Start command with optimized Gunicorn settings
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "300", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
