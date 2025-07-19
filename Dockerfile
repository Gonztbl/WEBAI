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
    dos2unix \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies (layer caching optimized)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# 4. Download and VERIFY model files with checksums (Corrected Syntax)
RUN mkdir -p model && \
    cd model && \
    \
    echo "Downloading fruit_state_classifier.keras..." && \
    wget -q "${MODEL_URL}/fruit_state_classifier.keras" && \
    \
    echo "Downloading yolo11n.pt..." && \
    wget -q "${MODEL_URL}/yolo11n.pt" && \
    \
    echo "Downloading fruit_ripeness_model_pytorch.pth..." && \
    wget -q "${MODEL_URL}/fruit_ripeness_model_pytorch.pth" && \
    \
    echo "Verifying all checksums..." && \
    cat <<EOF | dos2unix | sha256sum -c --strict
8ebe13c100c32f99911eb341e6b6278832a8848c909675239a587428803a6b5a3  fruit_state_classifier.keras
0ebbc80d4a7680d14987a577cd213c415555462589574163013a241e3d30925e  yolo11n.pt
48bf9333f4f07af2d02e3965f797f53f06b6b553e414c99736e4f165a6e87b7a6  fruit_ripeness_model_pytorch.pth
EOF

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
