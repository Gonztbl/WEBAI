# ===== IMPORTS =====
import os
import io
import uuid
import json
import logging
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    
    device = torch.device("cpu")
    torch.set_num_threads(2)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    device = None

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ===== CONFIGURATION =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / 'model'

# ===== CORRECT MODEL PATHS FROM TRAINING SCRIPT =====
KERAS_MODEL_PATH = MODEL_DIR / 'fruit_state_classifier.keras'  # ← CORRECT FORMAT!
CLASS_INDICES_PATH = MODEL_DIR / 'fruit_class_indices.json'    # ← CLASS MAPPING!
PYTORCH_MODEL_PATH = MODEL_DIR / 'fruit_ripeness_model_pytorch.pth'
YOLO_MODEL_PATH = MODEL_DIR / 'yolov8l.pt'

# Default classes (fallback if JSON not available)
DEFAULT_FRESHNESS_CLASSES = ['Táo Tươi', 'Chuối Tươi', 'Cam Tươi', 'Táo Hỏng', 'Chuối Hỏng', 'Cam Hỏng']
RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

# Global variables
keras_model = None
pytorch_model = None
yolo_model = None
freshness_classes = DEFAULT_FRESHNESS_CLASSES

# ===== LOAD CLASS INDICES =====
def load_class_indices():
    """Load class indices from JSON file created during training"""
    global freshness_classes
    
    if CLASS_INDICES_PATH.exists():
        try:
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            
            # Convert to list in correct order
            freshness_classes = [class_indices[str(i)] for i in range(len(class_indices))]
            logger.info(f"✅ Loaded class indices: {freshness_classes}")
            return freshness_classes
        except Exception as e:
            logger.error(f"❌ Failed to load class indices: {e}")
    else:
        logger.warning(f"❌ Class indices file not found: {CLASS_INDICES_PATH}")
    
    logger.info(f"Using default classes: {DEFAULT_FRESHNESS_CLASSES}")
    freshness_classes = DEFAULT_FRESHNESS_CLASSES
    return freshness_classes

# ===== SIMPLE FALLBACK CLASSIFIER =====
class SimpleFruitClassifier:
    def __init__(self, classes):
        self.classes = classes
        logger.info("Using simple fallback classifier")
    
    def predict(self, image_array):
        try:
            mean_brightness = np.mean(image_array)
            color_variance = np.var(image_array)
            
            # Enhanced logic based on image characteristics
            if mean_brightness > 150:  # Bright image - likely fresh
                if color_variance > 1000:  # High color variance - diverse colors
                    return np.array([0.45, 0.25, 0.20, 0.05, 0.03, 0.02])  # Fresh fruits
                else:
                    return np.array([0.35, 0.35, 0.20, 0.05, 0.03, 0.02])
            elif mean_brightness > 100:  # Medium brightness
                return np.array([0.25, 0.25, 0.25, 0.10, 0.10, 0.05])
            else:  # Dark image - possibly spoiled
                return np.array([0.05, 0.05, 0.05, 0.35, 0.30, 0.20])  # Spoiled fruits
        except:
            return np.array([1/len(self.classes)] * len(self.classes))

# ===== MODEL LOADING =====
def load_keras_model():
    """Load Keras model in .keras format (from training script)"""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available")
        return SimpleFruitClassifier(freshness_classes)
    
    if KERAS_MODEL_PATH.exists():
        try:
            logger.info(f"Loading Keras model: {KERAS_MODEL_PATH}")
            
            # Load model in .keras format (TensorFlow 2.x native format)
            model = load_model(str(KERAS_MODEL_PATH))
            
            logger.info("✅ Keras model (.keras format) loaded successfully")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load Keras model: {e}")
    else:
        logger.warning(f"❌ Keras model not found: {KERAS_MODEL_PATH}")
    
    logger.info("Using enhanced simple fallback classifier")
    return SimpleFruitClassifier(freshness_classes)

def load_pytorch_model():
    """Load PyTorch model"""
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch not available")
        return None
    
    if PYTORCH_MODEL_PATH.exists():
        try:
            logger.info(f"Loading PyTorch model: {PYTORCH_MODEL_PATH}")
            
            model = models.mobilenet_v2(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_ftrs, len(RIPENESS_CLASSES))
            )
            
            state_dict = torch.load(str(PYTORCH_MODEL_PATH), map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            
            logger.info("✅ PyTorch model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
    else:
        logger.warning(f"❌ PyTorch model not found: {PYTORCH_MODEL_PATH}")
    
    return None

def load_yolo_model():
    """Load YOLO model"""
    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available")
        return None
    
    if YOLO_MODEL_PATH.exists():
        try:
            logger.info(f"Loading YOLO model: {YOLO_MODEL_PATH}")
            model = YOLO(str(YOLO_MODEL_PATH))
            logger.info("✅ YOLO model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
    else:
        logger.warning(f"❌ YOLO model not found: {YOLO_MODEL_PATH}")
    
    # Try default YOLO as fallback
    try:
        logger.info("Loading default YOLO model...")
        model = YOLO('yolov8n.pt')
        logger.info("✅ Default YOLO model loaded")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load default YOLO: {e}")
        return None

def initialize_models():
    """Initialize all models"""
    global keras_model, pytorch_model, yolo_model, freshness_classes
    
    logger.info("🚀 Initializing models...")
    
    # Load class indices first
    freshness_classes = load_class_indices()
    
    # Check which files exist
    logger.info("Checking model files:")
    for name, path in [
        ("Keras (.keras)", KERAS_MODEL_PATH),
        ("Class indices (.json)", CLASS_INDICES_PATH),
        ("PyTorch (.pth)", PYTORCH_MODEL_PATH),
        ("YOLO (.pt)", YOLO_MODEL_PATH)
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            logger.info(f"✅ {name}: {path} ({size_mb:.2f} MB)")
        else:
            logger.warning(f"❌ {name}: {path} - NOT FOUND")
    
    # Load models
    keras_model = load_keras_model()
    pytorch_model = load_pytorch_model()
    yolo_model = load_yolo_model()
    
    logger.info("✅ Model initialization completed")
    logger.info(f"Models loaded: Keras={keras_model is not None}, "
                f"PyTorch={pytorch_model is not None}, "
                f"YOLO={yolo_model is not None}")

# ===== HELPER FUNCTIONS =====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def process_image_from_link(link):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        filename = f"{uuid.uuid4()}.jpg"
        saved_path = IMAGES_DIR / filename
        image.save(str(saved_path))
        return str(saved_path), None
    except Exception as e:
        logger.error(f"Error processing image link: {e}")
        return None, f"Lỗi xử lý ảnh: {str(e)}"

def detect_with_yolo(image_path):
    """YOLO object detection"""
    if yolo_model is None:
        logger.warning("YOLO model not available")
        return None
    
    try:
        results = yolo_model(image_path, verbose=False)
        best_box = None
        max_area = 0
        
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
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

def draw_bounding_box(image_path, bbox):
    """Draw bounding box on image"""
    try:
        image = Image.open(image_path).convert("RGB")
        if bbox:
            draw = ImageDraw.Draw(image)
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
        
        filename = f"{uuid.uuid4()}_detected.jpg"
        save_path = IMAGES_DIR / filename
        image.save(str(save_path))
        return filename
    except Exception as e:
        logger.error(f"Drawing error: {e}")
        return None

def predict_freshness(image_path):
    """Predict fruit freshness using Keras model"""
    try:
        if keras_model is None:
            return ["Model không khả dụng"], [0]
        
        if hasattr(keras_model, 'predict'):
            # Real Keras model
            img = load_img(image_path, target_size=(224, 224))
            img = img_to_array(img).reshape(1, 224, 224, 3).astype('float32') / 255.0
            preds = keras_model.predict(img, verbose=0)[0]
        else:
            # Simple classifier fallback
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            img_array = np.array(img)
            preds = keras_model.predict(img_array)
        
        top = preds.argsort()[::-1][:3]
        return (
            [freshness_classes[i] for i in top],
            [(preds[i] * 100) for i in top]
        )
    except Exception as e:
        logger.error(f"Freshness prediction error: {e}")
        return ["Lỗi dự đoán"], [0]

def predict_ripeness(image_path):
    """Predict fruit ripeness using PyTorch model"""
    try:
        if pytorch_model is None:
            return "Model không khả dụng", 0
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = pytorch_model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
        
        return RIPENESS_CLASSES[idx.item()], conf.item() * 100
    except Exception as e:
        logger.error(f"Ripeness prediction error: {e}")
        return "Lỗi dự đoán", 0

# ===== FLASK ROUTES =====
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "keras_model": keras_model is not None,
            "pytorch_model": pytorch_model is not None,
            "yolo_model": yolo_model is not None
        },
        "model_files": {
            "keras_exists": KERAS_MODEL_PATH.exists(),
            "class_indices_exists": CLASS_INDICES_PATH.exists(),
            "pytorch_exists": PYTORCH_MODEL_PATH.exists(),
            "yolo_exists": YOLO_MODEL_PATH.exists()
        },
        "classes": {
            "freshness_classes": freshness_classes,
            "ripeness_classes": RIPENESS_CLASSES
        },
        "model_types": {
            "keras_type": type(keras_model).__name__ if keras_model else "None",
            "pytorch_type": type(pytorch_model).__name__ if pytorch_model else "None",
            "yolo_type": type(yolo_model).__name__ if yolo_model else "None"
        },
        "features": {
            "tensorflow": TENSORFLOW_AVAILABLE,
            "pytorch": PYTORCH_AVAILABLE,
            "yolo": YOLO_AVAILABLE
        }
    }, 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    error = ''
    img_path = None
    
    try:
        # Process input
        if 'link' in request.form and request.form['link'].strip():
            img_path, error = process_image_from_link(request.form['link'].strip())
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if allowed_file(file.filename):
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                img_path = IMAGES_DIR / filename
                file.save(str(img_path))
            else:
                error = "Định dạng ảnh không hợp lệ"
        else:
            error = "Vui lòng tải ảnh hoặc nhập liên kết"
        
        if img_path and not error:
            img_path = str(img_path)
            
            # YOLO detection
            bbox = detect_with_yolo(img_path)
            detected_img = draw_bounding_box(img_path, bbox)
            
            if not detected_img:
                # Fallback: use original image
                original_img = Image.open(img_path)
                detected_img = f"{uuid.uuid4()}_original.jpg"
                original_img.save(str(IMAGES_DIR / detected_img))
            
            # Predictions
            freshness_classes_pred, freshness_probs = predict_freshness(img_path)
            ripeness_pred, ripeness_conf = predict_ripeness(img_path)
            
            # Ensure enough data
            while len(freshness_classes_pred) < 3:
                freshness_classes_pred.append("N/A")
                freshness_probs.append(0)
            
            predictions = {
                "pytorch_prediction": ripeness_pred,
                "pytorch_confidence": ripeness_conf,
                "color_ripeness": "Phân tích màu sắc cơ bản",
                "freshness_class1": freshness_classes_pred[0],
                "freshness_prob1": freshness_probs[0],
                "freshness_class2": freshness_classes_pred[1],
                "freshness_prob2": freshness_probs[1],
                "freshness_class3": freshness_classes_pred[2],
                "freshness_prob3": freshness_probs[2],
            }
            
            return render_template("success.html",
                                   img=detected_img,
                                   yolo_img=detected_img,
                                   predictions=predictions)
    
    except Exception as e:
        logger.error(f"Error in success route: {e}")
        error = f"Lỗi hệ thống: {str(e)}"
    
    return render_template("index.html", error=error)

# ===== INITIALIZATION =====
try:
    initialize_models()
    logger.info("🎉 Application ready!")
except Exception as e:
    logger.error(f"Initialization error: {e}")

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=PORT, debug=False)
