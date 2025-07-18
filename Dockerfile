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

# 2. Install Python dependencies (layer caching optimized)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# 4. Download model files with strict checks and retries
RUN mkdir -p model && \
    cd model && \
    for model in fruit_state_classifier.keras yolo11n.pt fruit_ripeness_model_pytorch.pth; do \
        echo "Downloading ${model}..."; \
        for i in {1..3}; do \
            wget -q --show-progress --progress=bar:force:noscroll \
            "${MODEL_URL}/${model}" && break || \
            { echo "Retry $i/3"; sleep 2; rm -f "${model}"; } \
        done && \
        [ -s "${model}" ] || { echo "ERROR: File empty"; exit 1; } && \
        case "${model}" in \
            "fruit_state_classifier.keras") checksum="a1b2c3d4e5f6...";; \
            "yolo11n.pt") checksum="x9y8z7...";; \
            "fruit_ripeness_model_pytorch.pth") checksum="p0o9i8...";; \
        esac && \
        (echo "${checksum} ${model}" | md5sum -c --strict - || \
            { echo "Checksum failed"; rm -f "${model}"; exit 1; }) \
    done && \
    # Clean potential duplicates
    rm -f *.1 *.2 && \
    echo "Model verification passed:" && \
    ls -lh

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
