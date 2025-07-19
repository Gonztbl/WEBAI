# ===== IMPORTS =====
import os
import io
import uuid
import json
import logging
import gc
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# TensorFlow imports with memory optimization
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    
    device = torch.device("cpu")
    torch.set_num_threads(1)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    device = None

# YOLO imports
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

# ===== CONFIGURATION =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / 'model'

# ===== CORRECT MODEL PATHS - SỬ DỤNG FILE .H5 CÓ SẴN =====
H5_MODEL_PATHS = [
    MODEL_DIR / 'fruit_state_classifier_new.h5',      # ← File bạn đã có!
    MODEL_DIR / 'fruit_state_classifier.h5',
    MODEL_DIR / 'fruit_state_classifier.keras'
]

CLASS_INDICES_PATH = MODEL_DIR / 'fruit_class_indices.json'
PYTORCH_MODEL_PATH = MODEL_DIR / 'fruit_ripeness_model_pytorch.pth'
YOLO_SMALL_PATH = MODEL_DIR / 'yolo11n.pt'
YOLO_LARGE_PATH = MODEL_DIR / 'yolov8l.pt'

DEFAULT_FRESHNESS_CLASSES = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

# Global variables
keras_model = None
pytorch_model = None
yolo_model = None
freshness_classes = DEFAULT_FRESHNESS_CLASSES

# ===== MEMORY MANAGEMENT =====
def clear_memory():
    """Clear memory and run garbage collection"""
    gc.collect()
    if TENSORFLOW_AVAILABLE:
        try:
            tf.keras.backend.clear_session()
        except:
            pass

def load_class_indices():
    """Load class indices from JSON file"""
    global freshness_classes
    
    if CLASS_INDICES_PATH.exists():
        try:
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            
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

# ===== ENHANCED FALLBACK CLASSIFIER =====
class EnhancedFruitClassifier:
    def __init__(self, classes):
        self.classes = classes
        logger.info("Using enhanced fallback classifier")
    
    def predict(self, image_array):
        try:
            mean_brightness = np.mean(image_array)
            color_variance = np.var(image_array)
            
            if len(image_array.shape) == 3:
                r_mean = np.mean(image_array[:, :, 0])
                g_mean = np.mean(image_array[:, :, 1])
                b_mean = np.mean(image_array[:, :, 2])
                
                # Enhanced color-based detection
                if r_mean > g_mean and r_mean > b_mean:  # Red - apple
                    if mean_brightness > 120:
                        return np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02])  # Fresh apple
                    else:
                        return np.array([0.1, 0.05, 0.05, 0.6, 0.15, 0.05])  # Rotten apple
                elif (r_mean + g_mean) > b_mean * 1.5:  # Yellow/orange
                    if mean_brightness > 130:
                        return np.array([0.1, 0.4, 0.3, 0.05, 0.1, 0.05])  # Fresh banana/orange
                    else:
                        return np.array([0.05, 0.1, 0.1, 0.1, 0.4, 0.25])  # Rotten banana/orange
            
            # Fallback based on brightness
            if mean_brightness > 150:
                return np.array([0.35, 0.25, 0.25, 0.05, 0.05, 0.05])
            elif mean_brightness > 100:
                return np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
            else:
                return np.array([0.05, 0.05, 0.05, 0.3, 0.3, 0.25])
                
        except Exception as e:
            logger.error(f"Enhanced classifier error: {e}")
            return np.array([1/len(self.classes)] * len(self.classes))

# ===== MODEL LOADING - PRIORITIZE .H5 FILES =====
def load_keras_model():
    """Load Keras model - prioritize .h5 files"""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available")
        return EnhancedFruitClassifier(freshness_classes)
    
    # Try loading existing .h5 files first
    for model_path in H5_MODEL_PATHS:
        if model_path.exists():
            try:
                logger.info(f"Trying to load H5 model: {model_path}")
                
                # Multiple loading strategies for .h5 files
                loading_strategies = [
                    lambda: load_model(str(model_path)),
                    lambda: load_model(str(model_path), compile=False),
                    lambda: tf.keras.models.load_model(str(model_path), compile=False)
                ]
                
                for i, strategy in enumerate(loading_strategies):
                    try:
                        model = strategy()
                        
                        # Recompile if needed
                        if not hasattr(model, 'optimizer') or model.optimizer is None:
                            model.compile(
                                optimizer='adam',
                                loss='categorical_crossentropy',
                                metrics=['accuracy']
                            )
                        
                        logger.info(f"✅ H5 model loaded successfully with strategy {i+1}")
                        logger.info(f"Model input shape: {model.input_shape}")
                        logger.info(f"Model output shape: {model.output_shape}")
                        return model
                        
                    except Exception as e:
                        logger.warning(f"Strategy {i+1} failed: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Failed to load {model_path}: {e}")
                continue
    
    logger.info("Using enhanced fallback classifier")
    return EnhancedFruitClassifier(freshness_classes)

def load_pytorch_model():
    """Load PyTorch model with memory optimization"""
    if not PYTORCH_AVAILABLE:
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
            
            del state_dict
            clear_memory()
            
            logger.info("✅ PyTorch model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
    
    return None

def load_yolo_model_on_demand():
    """Load YOLO model only when needed"""
    global yolo_model
    
    if yolo_model is not None:
        return yolo_model
    
    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available")
        return None
    
    # Try smaller model first to save memory
    model_paths = [
        (YOLO_SMALL_PATH, "small YOLO"),
        (YOLO_LARGE_PATH, "large YOLO")
    ]
    
    for model_path, name in model_paths:
        if model_path.exists():
            try:
                logger.info(f"Loading {name}: {model_path}")
                yolo_model = YOLO(str(model_path))
                logger.info(f"✅ {name} loaded successfully")
                return yolo_model
            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
                clear_memory()
    
    # Try default small model
    try:
        logger.info("Loading default small YOLO...")
        yolo_model = YOLO('yolov8n.pt')
        logger.info("✅ Default small YOLO loaded")
        return yolo_model
    except Exception as e:
        logger.error(f"❌ Failed to load default YOLO: {e}")
        return None

def initialize_models():
    """Initialize models with memory optimization"""
    global keras_model, pytorch_model
    
    logger.info("🚀 Initializing models (prioritizing .h5 format)...")
    
    # Load class indices first
    freshness_classes = load_class_indices()
    
    # Check which files exist
    logger.info("Checking model files:")
    for name, path in [
        ("H5 model (new)", H5_MODEL_PATHS[0]),
        ("H5 model (old)", H5_MODEL_PATHS[1]),
        ("Keras model", H5_MODEL_PATHS[2]),
        ("Class indices (.json)", CLASS_INDICES_PATH),
        ("PyTorch (.pth)", PYTORCH_MODEL_PATH),
        ("YOLO small (.pt)", YOLO_SMALL_PATH),
        ("YOLO large (.pt)", YOLO_LARGE_PATH)
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            logger.info(f"✅ {name}: {size_mb:.2f} MB")
        else:
            logger.warning(f"❌ {name}: NOT FOUND")
    
    # Load models sequentially
    try:
        keras_model = load_keras_model()
        clear_memory()
        
        pytorch_model = load_pytorch_model()
        clear_memory()
        
        # YOLO will be loaded on demand
        logger.info("YOLO model will be loaded on demand")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        clear_memory()
    
    logger.info("✅ Model initialization completed")
    logger.info(f"Models loaded: Keras={keras_model is not None}, "
                f"PyTorch={pytorch_model is not None}")

# ===== HELPER FUNCTIONS =====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def process_image_from_link(link):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        if image.size[0] > 1024 or image.size[1] > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        filename = f"{uuid.uuid4()}.jpg"
        saved_path = IMAGES_DIR / filename
        image.save(str(saved_path), quality=85)
        return str(saved_path), None
    except Exception as e:
        logger.error(f"Error processing image link: {e}")
        return None, f"Lỗi xử lý ảnh: {str(e)}"

def detect_with_yolo(image_path):
    """YOLO detection with on-demand loading"""
    try:
        yolo = load_yolo_model_on_demand()
        if yolo is None:
            logger.warning("YOLO model not available")
            return None
        
        results = yolo(image_path, verbose=False)
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
        
        clear_memory()
        return best_box
        
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        clear_memory()
        return None

def draw_bounding_box(image_path, bbox):
    """Draw bounding box with memory optimization"""
    try:
        image = Image.open(image_path).convert("RGB")
        if bbox:
            draw = ImageDraw.Draw(image)
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        filename = f"{uuid.uuid4()}_detected.jpg"
        save_path = IMAGES_DIR / filename
        image.save(str(save_path), quality=85)
        return filename
    except Exception as e:
        logger.error(f"Drawing error: {e}")
        return None

def predict_freshness(image_path):
    """Predict freshness with memory optimization"""
    try:
        if keras_model is None:
            return ["Model không khả dụng"], [0]
        
        if hasattr(keras_model, 'predict'):
            # Real Keras model
            img = load_img(image_path, target_size=(224, 224))
            img = img_to_array(img).reshape(1, 224, 224, 3).astype('float32') / 255.0
            preds = keras_model.predict(img, verbose=0)[0]
        else:
            # Enhanced classifier
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            img_array = np.array(img)
            preds = keras_model.predict(img_array)
        
        top = preds.argsort()[::-1][:3]
        result = (
            [freshness_classes[i] for i in top],
            [(preds[i] * 100) for i in top]
        )
        
        clear_memory()
        return result
        
    except Exception as e:
        logger.error(f"Freshness prediction error: {e}")
        return ["Lỗi dự đoán"], [0]

def predict_ripeness(image_path):
    """Predict ripeness with memory optimization"""
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
        
        result = RIPENESS_CLASSES[idx.item()], conf.item() * 100
        
        del input_tensor, output, probs
        clear_memory()
        
        return result
        
    except Exception as e:
        logger.error(f"Ripeness prediction error: {e}")
        return "Lỗi dự đoán", 0

# ===== FLASK ROUTES =====
@app.route('/health')
def health_check():
    """Health check with model info"""
    return {
        "status": "healthy",
        "models": {
            "keras_model": keras_model is not None,
            "keras_type": type(keras_model).__name__ if keras_model else "None",
            "pytorch_model": pytorch_model is not None,
            "yolo_available": YOLO_AVAILABLE
        },
        "model_files": {
            "h5_new_exists": H5_MODEL_PATHS[0].exists(),
            "h5_old_exists": H5_MODEL_PATHS[1].exists(),
            "keras_exists": H5_MODEL_PATHS[2].exists(),
            "class_indices_exists": CLASS_INDICES_PATH.exists(),
            "pytorch_exists": PYTORCH_MODEL_PATH.exists()
        },
        "classes": {
            "freshness_classes": freshness_classes,
            "total_classes": len(freshness_classes)
        },
        "memory_optimized": True
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
            
            # YOLO detection (on-demand)
            bbox = detect_with_yolo(img_path)
            detected_img = draw_bounding_box(img_path, bbox)
            
            if not detected_img:
                original_img = Image.open(img_path)
                detected_img = f"{uuid.uuid4()}_original.jpg"
                original_img.save(str(IMAGES_DIR / detected_img), quality=85)
            
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
                "color_ripeness": "Phân tích màu sắc nâng cao",
                "freshness_class1": freshness_classes_pred[0],
                "freshness_prob1": freshness_probs[0],
                "freshness_class2": freshness_classes_pred[1],
                "freshness_prob2": freshness_probs[1],
                "freshness_class3": freshness_classes_pred[2],
                "freshness_prob3": freshness_probs[2],
            }
            
            clear_memory()
            
            return render_template("success.html",
                                   img=detected_img,
                                   yolo_img=detected_img,
                                   predictions=predictions)
    
    except Exception as e:
        logger.error(f"Error in success route: {e}")
        error = f"Lỗi hệ thống: {str(e)}"
        clear_memory()
    
    return render_template("index.html", error=error)

# ===== INITIALIZATION =====
try:
    initialize_models()
    logger.info("🎉 H5-optimized application ready!")
except Exception as e:
    logger.error(f"Initialization error: {e}")
    clear_memory()

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=PORT, debug=False)
