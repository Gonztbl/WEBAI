# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

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

# 4. Download model files with strict checks
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
    # Remove any accidental duplicates
    rm -f *.1 *.2

# 5. Create directories for static files
RUN mkdir -p static/images && \
    chmod -R a+rwx static

# 6. Health check configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# 7. Expose port
EXPOSE 10000

# 8. Run as non-root user for security
RUN useradd -m appuser && chown -R appuser /app
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
