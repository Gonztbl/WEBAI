# ===== PHẦN IMPORT ĐẦY ĐỦ VÀ SẠCH SẼ =====
import os
import io
import uuid
import urllib.request
import logging
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# TensorFlow imports với error handling
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    
    # Tắt GPU và giảm log spam cho TensorFlow CPU
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    TENSORFLOW_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TensorFlow CPU loaded successfully")
except ImportError as e:
    logging.warning(f"TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    
    # Force CPU usage for PyTorch
    device = torch.device("cpu")
    torch.set_num_threads(2)
    
    PYTORCH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("PyTorch CPU loaded successfully")
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    PYTORCH_AVAILABLE = False
    device = None

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("YOLO loaded successfully")
except ImportError as e:
    logging.warning(f"YOLO not available: {e}")
    YOLO_AVAILABLE = False

# Color analysis
try:
    import webcolors
    import cv2
    from sklearn.cluster import KMeans
    COLOR_ANALYSIS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Color analysis tools loaded successfully")
except ImportError as e:
    logging.warning(f"Color analysis not available: {e}")
    COLOR_ANALYSIS_AVAILABLE = False

# ===== CẤU HÌNH CƠ BẢN =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CẤU HÌNH ĐƯỜNG DẪN =====
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / 'model'

# Đường dẫn model - CẬP NHẬT với tên file chính xác
CLASSIFIER_MODEL_PATH = MODEL_DIR / 'fruit_state_classifier.weights.h5'
CLASSIFIER_FULL_MODEL_PATH = MODEL_DIR / 'fruit_state_classifier.h5'
CLASSIFIER_KERAS_MODEL_PATH = MODEL_DIR / 'fruit_state_classifier.keras'
DETECTOR_MODEL_PATH = MODEL_DIR / 'yolov8l.pt'
RIPENESS_MODEL_PATH = MODEL_DIR / 'fruit_ripeness_model_pytorch.pth'

# Các classes dự đoán
FRESHNESS_CLASSES = ['Táo Tươi', 'Chuối Tươi', 'Cam Tươi', 'Táo Hỏng', 'Chuối Hỏng', 'Cam Hỏng']
RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
NUM_PYTORCH_CLASSES = len(RIPENESS_CLASSES)
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

# Global model variables
classifier_model = None
detector_model = None
ripeness_model = None

# ===== EXACT MODEL ARCHITECTURE =====
def create_exact_model_architecture():
    """Tạo chính xác architecture như trong model summary"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        # Tạo base model MobileNetV2 chính xác như trong summary
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Tạo Sequential model với chính xác 5 layers như summary
        model = Sequential([
            base_model,                           # Layer 1: mobilenetv2_1.00_224
            GlobalAveragePooling2D(),            # Layer 2: global_average_pooling2d
            Dropout(0.2),                        # Layer 3: dropout
            Dense(128, activation='relu'),       # Layer 4: dense
            Dense(6, activation='softmax')       # Layer 5: dense_1 (6 classes)
        ])
        
        logger.info("Created exact model architecture matching summary")
        logger.info(f"Model layers: {len(model.layers)}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create exact model architecture: {e}")
        return None

# ===== SIMPLE FALLBACK CLASSIFIER =====
class SimpleFruitClassifier:
    """Classifier dự phòng đơn giản"""
    
    def __init__(self):
        self.classes = FRESHNESS_CLASSES
        logger.info("Initialized simple fallback classifier")
    
    def predict(self, image_array):
        """Dự đoán đơn giản dựa trên đặc trưng cơ bản"""
        try:
            # Tính toán các đặc trưng cơ bản
            mean_brightness = np.mean(image_array)
            color_variance = np.var(image_array)
            
            # Logic đơn giản để phân loại
            if mean_brightness > 150:  # Ảnh sáng - có thể là trái cây tươi
                if color_variance > 1000:  # Màu sắc đa dạng
                    predictions = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
                else:
                    predictions = [0.3, 0.4, 0.2, 0.05, 0.03, 0.02]
            else:  # Ảnh tối - có thể là trái cây hỏng
                predictions = [0.1, 0.1, 0.1, 0.3, 0.2, 0.2]
            
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Error in simple classifier: {e}")
            return np.array([1/6] * 6)

# ===== MODEL LOADING FUNCTIONS =====
def verify_models():
    """Kiểm tra tất cả model files"""
    model_status = {}
    
    model_files = [
        ("classifier_weights", CLASSIFIER_MODEL_PATH),
        ("classifier_full", CLASSIFIER_FULL_MODEL_PATH),
        ("classifier_keras", CLASSIFIER_KERAS_MODEL_PATH),
        ("detector", DETECTOR_MODEL_PATH), 
        ("ripeness", RIPENESS_MODEL_PATH)
    ]
    
    for name, model_path in model_files:
        if model_path.exists() and model_path.stat().st_size > 0:
            size_mb = model_path.stat().st_size / 1024 / 1024
            logger.info(f"Model {name} found: {model_path} ({size_mb:.2f} MB)")
            model_status[name] = True
        else:
            logger.warning(f"Model {name} not found: {model_path}")
            model_status[name] = False
    
    return model_status

def load_keras_classifier():
    """Load Keras classifier với exact architecture"""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, using simple classifier")
        return SimpleFruitClassifier()
    
    try:
        # Phương pháp 1: Thử load full model (.h5 hoặc .keras)
        model_files_to_try = [
            CLASSIFIER_KERAS_MODEL_PATH,  # .keras file (preferred)
            CLASSIFIER_FULL_MODEL_PATH,   # .h5 full model
        ]
        
        for model_file in model_files_to_try:
            if model_file.exists():
                try:
                    logger.info(f"Loading full model: {model_file}")
                    model = load_model(str(model_file))
                    logger.info("Successfully loaded full Keras model")
                    logger.info(f"Model summary: {model.summary()}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load {model_file}: {e}")
        
        # Phương pháp 2: Tạo exact architecture và load weights
        if CLASSIFIER_MODEL_PATH.exists():
            logger.info("Creating exact architecture and loading weights...")
            
            model = create_exact_model_architecture()
            if model is not None:
                try:
                    # Load weights với exact architecture
                    model.load_weights(str(CLASSIFIER_MODEL_PATH))
                    logger.info("Successfully loaded weights with exact architecture")
                    logger.info(f"Loaded model layers: {len(model.layers)}")
                    return model
                except Exception as e:
                    logger.error(f"Failed to load weights with exact architecture: {e}")
        
        # Phương pháp 3: Tạo model mới với pre-trained weights
        logger.info("Creating new pre-trained model...")
        model = create_exact_model_architecture()
        if model is not None:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Created new pre-trained model (no custom weights)")
            return model
        
    except Exception as e:
        logger.error(f"All Keras loading methods failed: {e}")
    
    # Fallback cuối cùng
    logger.info("Using simple fallback classifier")
    return SimpleFruitClassifier()

def load_yolo_detector():
    """Load YOLO detector"""
    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available")
        return None
    
    try:
        if DETECTOR_MODEL_PATH.exists():
            logger.info("Loading custom YOLO model...")
            model = YOLO(str(DETECTOR_MODEL_PATH))
            logger.info("Custom YOLO model loaded successfully")
            return model
    except Exception as e:
        logger.error(f"Failed to load custom YOLO: {e}")
    
    try:
        logger.info("Loading default YOLO model...")
        model = YOLO('yolov8n.pt')  # Smaller model
        logger.info("Default YOLO model loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load default YOLO: {e}")
        return None

def load_pytorch_ripeness():
    """Load PyTorch ripeness model"""
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch not available")
        return None
    
    try:
        if RIPENESS_MODEL_PATH.exists():
            logger.info("Loading custom PyTorch model...")
            model = models.mobilenet_v2(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_ftrs, NUM_PYTORCH_CLASSES)
            )
            
            # Load state dict với map_location cho CPU
            state_dict = torch.load(str(RIPENESS_MODEL_PATH), map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            
            logger.info("Custom PyTorch model loaded successfully")
            return model
    except Exception as e:
        logger.error(f"Failed to load custom PyTorch model: {e}")
    
    try:
        logger.info("Creating default PyTorch model...")
        model = models.mobilenet_v2(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, NUM_PYTORCH_CLASSES)
        )
        model = model.to(device).eval()
        logger.info("Default PyTorch model created")
        return model
    except Exception as e:
        logger.error(f"Failed to create PyTorch model: {e}")
        return None

def initialize_models():
    """Khởi tạo tất cả models"""
    global classifier_model, detector_model, ripeness_model
    
    logger.info("Starting model initialization...")
    
    # Kiểm tra models
    model_status = verify_models()
    
    # Load classifier
    try:
        classifier_model = load_keras_classifier()
        logger.info(f"Classifier type: {type(classifier_model).__name__}")
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        classifier_model = SimpleFruitClassifier()
    
    # Load detector
    try:
        detector_model = load_yolo_detector()
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        detector_model = None
    
    # Load ripeness model
    try:
        ripeness_model = load_pytorch_ripeness()
    except Exception as e:
        logger.error(f"Failed to load ripeness model: {e}")
        ripeness_model = None
    
    logger.info("Model initialization completed")
    logger.info(f"Available models: Classifier={classifier_model is not None}, "
                f"Detector={detector_model is not None}, "
                f"Ripeness={ripeness_model is not None}")

# ===== COLOR ANALYSIS FUNCTIONS =====
def get_dominant_colors(image_path, k=5):
    """Lấy màu chủ đạo từ ảnh sử dụng K-means"""
    if not COLOR_ANALYSIS_AVAILABLE:
        return [], None
    
    try:
        # Đọc ảnh
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape ảnh thành array 2D
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Áp dụng K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Chuyển đổi centers về int
        centers = np.uint8(centers)
        
        # Tạo ảnh với màu chủ đạo
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        
        # Lưu ảnh kết quả
        result_name = f"{uuid.uuid4()}_colors.jpg"
        result_path = IMAGES_DIR / result_name
        
        segmented_pil = Image.fromarray(segmented_image)
        segmented_pil.save(str(result_path))
        
        return centers.tolist(), result_name
        
    except Exception as e:
        logger.error(f"Error in color analysis: {e}")
        return [], None

def classify_color_ripeness(dominant_colors):
    """Phân loại độ chín dựa trên màu sắc"""
    if not dominant_colors:
        return "Không xác định"
    
    try:
        # Tính toán các đặc trưng màu
        avg_color = np.mean(dominant_colors, axis=0)
        r, g, b = avg_color
        
        # Logic đơn giản để phân loại
        if r > 150 and g < 100 and b < 100:  # Đỏ - có thể chín
            return "Chín (màu đỏ)"
        elif r > 200 and g > 150 and b < 100:  # Vàng - có thể chín
            return "Chín (màu vàng)"
        elif g > 150 and r < 150 and b < 150:  # Xanh - chưa chín
            return "Chưa chín (màu xanh)"
        elif r < 100 and g < 100 and b < 100:  # Tối - có thể hỏng
            return "Có thể hỏng (màu tối)"
        else:
            return "Trung bình"
            
    except Exception as e:
        logger.error(f"Error in color classification: {e}")
        return "Không xác định"

# ==================== HÀM HỖ TRỢ ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def process_image_from_link(link):
    """Xử lý ảnh từ link với requests"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(link, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        filename = f"{uuid.uuid4()}.jpg"
        saved_path = IMAGES_DIR / filename
        image.save(str(saved_path))
        return str(saved_path), None
    except Exception as e:
        logger.error(f"Error processing image link: {e}")
        return None, "Lỗi khi xử lý liên kết ảnh"

def detect_fruit_with_yolo(image_path):
    """YOLO detection với error handling"""
    try:
        if detector_model is None:
            logger.warning("YOLO model not available")
            return None
        
        results = detector_model(image_path, verbose=False)
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

def draw_yolo_bounding_box(image_path, bounding_box):
    """Vẽ bounding box"""
    try:
        image = Image.open(image_path).convert("RGB")
        if bounding_box:
            draw = ImageDraw.Draw(image)
            x, y, w, h = bounding_box
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
        
        saved_name = f"{uuid.uuid4()}_yolo.jpg"
        save_path = IMAGES_DIR / saved_name
        image.save(str(save_path))
        return saved_name
    except Exception as e:
        logger.error(f"Error drawing bounding box: {e}")
        return None

def predict_fruit_freshness(image_path):
    """Dự đoán độ tươi của trái cây"""
    try:
        if classifier_model is None:
            return ["Model không khả dụng"], [0]
        
        if hasattr(classifier_model, 'predict'):
            # Keras model
            img = load_img(image_path, target_size=(224, 224))
            img = img_to_array(img).reshape(1, 224, 224, 3).astype('float32') / 255.0
            preds = classifier_model.predict(img, verbose=0)[0]
        else:
            # Simple classifier
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            preds = classifier_model.predict(img_array)
        
        top = preds.argsort()[::-1][:3]
        return (
            [FRESHNESS_CLASSES[i] for i in top],
            [(preds[i] * 100) for i in top]
        )
    except Exception as e:
        logger.error(f"Freshness prediction error: {e}")
        return ["Lỗi dự đoán"], [0]

def predict_ripeness_pytorch(image_path):
    """Dự đoán độ chín với PyTorch"""
    try:
        if ripeness_model is None:
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
            output = ripeness_model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
        
        return RIPENESS_CLASSES[idx.item()], conf.item() * 100
    except Exception as e:
        logger.error(f"Ripeness prediction error: {e}")
        return "Lỗi dự đoán", 0

def analyze_color(image_path, bounding_box=None):
    """Phân tích màu sắc"""
    try:
        dominant_colors, color_img = get_dominant_colors(image_path)
        color_ripeness = classify_color_ripeness(dominant_colors)
        return color_ripeness, color_img
    except Exception as e:
        logger.error(f"Color analysis error: {e}")
        
        # Fallback: copy ảnh gốc
        try:
            original_img = Image.open(image_path)
            fallback_name = f"{uuid.uuid4()}_original.jpg"
            original_img.save(str(IMAGES_DIR / fallback_name))
            return "Không xác định", fallback_name
        except:
            return "Không xác định", None

# ==================== FLASK ROUTES ====================
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "classifier": classifier_model is not None,
            "detector": detector_model is not None,
            "ripeness": ripeness_model is not None
        },
        "features": {
            "tensorflow": TENSORFLOW_AVAILABLE,
            "pytorch": PYTORCH_AVAILABLE,
            "yolo": YOLO_AVAILABLE,
            "color_analysis": COLOR_ANALYSIS_AVAILABLE
        },
        "device": str(device) if device else "None",
        "classifier_type": type(classifier_model).__name__ if classifier_model else "None"
    }, 200

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
                img_path = IMAGES_DIR / filename
                file.save(str(img_path))
            else:
                error = "Định dạng ảnh không hợp lệ"
        else:
            error = "Vui lòng tải ảnh hoặc nhập liên kết"
        
        # Xử lý ảnh nếu có
        if img_path and not error:
            img_path = str(img_path)
            
            # YOLO detection
            bbox = detect_fruit_with_yolo(img_path)
            yolo_img = draw_yolo_bounding_box(img_path, bbox)
            
            if not yolo_img:
                # Fallback: sử dụng ảnh gốc
                original_img = Image.open(img_path)
                yolo_img = f"{uuid.uuid4()}_original.jpg"
                original_img.save(str(IMAGES_DIR / yolo_img))
            
            # Predictions
            freshness_classes, freshness_probs = predict_fruit_freshness(img_path)
            color_ripeness, kmean_img = analyze_color(img_path, bbox)
            ripeness_pred, ripeness_conf = predict_ripeness_pytorch(img_path)
            
            # Đảm bảo có đủ dữ liệu
            while len(freshness_classes) < 3:
                freshness_classes.append("N/A")
                freshness_probs.append(0)
            
            if not kmean_img:
                kmean_img = yolo_img
            
            # Chuẩn bị dictionary cho template
            predictions_for_template = {
                "pytorch_prediction": ripeness_pred,
                "pytorch_confidence": ripeness_conf,
                "color_ripeness": color_ripeness,
                "freshness_class1": freshness_classes[0],
                "freshness_prob1": freshness_probs[0],
                "freshness_class2": freshness_classes[1],
                "freshness_prob2": freshness_probs[1],
                "freshness_class3": freshness_classes[2],
                "freshness_prob3": freshness_probs[2],
            }
            
            return render_template("success.html",
                                   img=kmean_img,
                                   yolo_img=yolo_img,
                                   predictions=predictions_for_template)
    
    except Exception as e:
        logger.error(f"Error in /success: {e}", exc_info=True)
        error = f"Lỗi hệ thống: {str(e)}"
    
    return render_template("index.html", error=error)

# ===== KHỞI TẠO ỨNG DỤNG =====
try:
    initialize_models()
    logger.info("Application started successfully")
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=PORT, debug=False)
