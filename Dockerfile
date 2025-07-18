# Step 1: Use an official Python Docker image as a base
FROM python:3.10-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Update and install wget and necessary libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Step 4: Copy the source code and other files
COPY . .

# Step 5: Create the model directory
RUN mkdir -p model

# Step 6: Download large model files directly from GitHub Releases
RUN wget -O model/fruit_state_classifier.keras "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/fruit_state_classifier.keras"
RUN wget -O model/yolo11n.pt "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/yolo11n.pt"
RUN wget -O model/fruit_ripeness_model_pytorch.pth "https://github.com/Gonztbl/WEBAI/releases/download/v.1.1/fruit_ripeness_model_pytorch.pth"

# Step 7: Install Python libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 8: Expose port 10000
EXPOSE 10000

# Step 9: Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "300", "--preload", "app:app"]
