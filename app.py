# ===== PHẦN IMPORT ĐẦY ĐỦ =====
import os
import io
import uuid
import urllib.request
import logging
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import models, transforms
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from dominant_color import color_of_image, name_main_color
from ultralytics import YOLO

# ===== CẤU HÌNH CƠ BẢN =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thiết bị PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== CẤU HÌNH ĐƯỜNG DẪN =====
BASE_DIR = Path(__file__).parent
IMAGES_DIR = Path("/tmp/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR/'model'

# Tạo thư mục nếu chưa tồn tại
IMAGES_DIR.mkdir(parents=True, exist_ok=True)



# Đường dẫn model
CLASSIFIER_MODEL_PATH = MODEL_DIR/'fruit_state_classifier.keras'
DETECTOR_MODEL_PATH = MODEL_DIR/'yolo11n.pt'
RIPENESS_MODEL_PATH = MODEL_DIR/'fruit_ripeness_model_pytorch.pth'

# Các classes dự đoán
FRESHNESS_CLASSES = ['Táo Tươi', 'Chuối Tươi', 'Cam Tươi', 'Táo Hỏng', 'Chuối Hỏng', 'Cam Hỏng']
RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
NUM_PYTORCH_CLASSES = len(RIPENESS_CLASSES)
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

# ===== KIỂM TRA VÀ LOAD MODEL =====
def verify_models():
    """Kiểm tra tất cả model files tồn tại và hợp lệ"""
    for model_path in [CLASSIFIER_MODEL_PATH, DETECTOR_MODEL_PATH, RIPENESS_MODEL_PATH]:
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_path.stat().st_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            raise ValueError(f"Model file is empty: {model_path}")
        
        logger.info(f"Model verified: {model_path} ({model_path.stat().st_size/1024/1024:.2f} MB)")

def load_models():
    """Load các model với xử lý lỗi chi tiết"""
    models = {}
    
    try:
        logger.info("Loading Keras model...")
        models['classifier'] = load_model(str(CLASSIFIER_MODEL_PATH))
        logger.info("Keras model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Keras model: {e}")
        raise
    
    try:
        logger.info("Loading YOLO model...")
        models['detector'] = YOLO(str(DETECTOR_MODEL_PATH))
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise
    
    try:
        logger.info("Loading PyTorch model...")
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(num_ftrs, NUM_PYTORCH_CLASSES)
        )
        model.load_state_dict(torch.load(str(RIPENESS_MODEL_PATH), map_location=device))
        models['ripeness'] = model.to(device).eval()
        logger.info("PyTorch model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        raise
    
    return models

# Khởi tạo ứng dụng
try:
    verify_models()
    model_dict = load_models()
    classifier_model = model_dict['classifier']
    detector_model = model_dict['detector']
    ripeness_model = model_dict['ripeness']
except Exception as e:
    logger.critical(f"Failed to initialize models: {e}")
    raise

# ==================== HÀM HỖ TRỢ ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def process_image_from_link(link):
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        with urllib.request.urlopen(link, timeout=10) as response:
            if response.status != 200:
                return None, "Không thể tải ảnh từ liên kết"
                
            image_bytes = response.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            filename = f"{uuid.uuid4()}.jpg"
            saved_path = IMAGES_DIR/filename
            image.save(str(save_path))
            return str(saved_path), None
    except Exception as e:
        logger.error(f"Error processing image link: {e}")
        return None, "Lỗi khi xử lý liên kết ảnh"

def detect_fruit_with_yolo(image_path):
    try:
        results = detector_model(image_path, verbose=False)
        best_box = None
        max_area = 0
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                area = w * h
                if area > max_area:
                    max_area = area
                    best_box = (x, y, w, h)
        return best_box
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        return None

def draw_yolo_bounding_box(image_path, bounding_box):
    try:
        image = Image.open(image_path).convert("RGB")
        if bounding_box:
            draw = ImageDraw.Draw(image)
            x, y, w, h = bounding_box
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
        
        saved_name = f"{uuid.uuid4()}_yolo.jpg"
        save_path = IMAGES_DIR/saved_name
        image.save(str(save_path))
        return saved_name
    except Exception as e:
        logger.error(f"Error drawing bounding box: {e}")
        return None

def predict_fruit_freshness(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img).reshape(1, 224, 224, 3).astype('float32') / 255.0
        preds = classifier_model.predict(img, verbose=0)[0]
        top = preds.argsort()[::-1][:3]
        return (
            [FRESHNESS_CLASSES[i] for i in top],
            [(preds[i]*100).round(2) for i in top]
        )
    except Exception as e:
        logger.error(f"Freshness prediction error: {e}")
        return ["Lỗi dự đoán"], [0]

def predict_ripeness_pytorch(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = ripeness_model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
        
        return RIPENESS_CLASSES[idx.item()], conf.item() * 100
    except Exception as e:
        logger.error(f"Ripeness prediction error: {e}")
        return "Lỗi dự đoán", 0

# ==================== FLASK ROUTES ====================
@app.route('/health')
def health_check():
    return {"status": "healthy"}, 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    error = ''
    img_path = None
    
    try:
        # Xử lý đầu vào
        if 'link' in request.form and request.form['link'].strip():
            img_path, error = process_image_from_link(request.form['link'].strip())
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if allowed_file(file.filename):
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                img_path = IMAGES_DIR/filename
                image.save(str(save_path))
            else:
                error = "Định dạng ảnh không hợp lệ"
        else:
            error = "Vui lòng tải ảnh hoặc nhập liên kết"

        # Xử lý ảnh nếu có
        if img_path and not error:
            img_path = str(img_path)
            bbox = detect_fruit_with_yolo(img_path)
            yolo_img = draw_yolo_bounding_box(img_path, bbox)
            
            if not yolo_img:
                raise ValueError("Không thể vẽ bounding box")
            
            freshness_classes, freshness_probs = predict_fruit_freshness(img_path)
            dominant_colors, kmean_img = color_of_image(img_path, bounding_box=bbox)
            color_ripeness = name_main_color(dominant_colors)
            ripeness_pred, ripeness_conf = predict_ripeness_pytorch(img_path)

            return render_template("success.html",
                img=kmean_img,
                yolo_img=yolo_img,
                predictions={
                    "freshness": list(zip(freshness_classes, freshness_probs)),
                    "color_ripeness": color_ripeness,
                    "ripeness_pred": ripeness_pred,
                    "ripeness_conf": ripeness_conf
                })
    
    except Exception as e:
        logger.error(f"Error in /success: {e}", exc_info=True)
        error = f"Lỗi hệ thống: {str(e)}"
    
    return render_template("index.html", error=error)

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=PORT, debug=False)
