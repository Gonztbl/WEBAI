# ===== FINAL SYNTHESIZED & FIXED VERSION =====
import os
import io
import uuid
import urllib.request
import json
import logging
import base64
import gc
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===== ROBUST LIBRARY IMPORTS =====
# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
# OpenCV import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ==================== CONFIGURATION (MOVED UP) ====================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    
    # --- Model Paths ---
    KERAS_MODEL_PATHS = [
        os.path.join(MODEL_DIR, 'fruit_state_classifier.keras'),
        os.path.join(MODEL_DIR, 'fruit_state_classifier_final.h5'),
    ]
    CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'fruit_class_indices.json')
    DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'yolo11n.pt')
    RIPENESS_MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_ripeness_model_pytorch.pth')

    # --- Classes ---
    FRESHNESS_CLASSES_FALLBACK = ['Táo Tươi', 'Chuối Tươi', 'Cam Tươi', 'Táo Hỏng', 'Chuối Hỏng', 'Cam Hỏng']
    RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
    
    # --- File Settings & Processing ---
    ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif', 'webp'}
    MAX_FILE_SIZE = 10 * 1024 * 1024
    TARGET_SIZE = (224, 224)
    CONFIDENCE_THRESHOLD = 0.5
    ENSEMBLE_WEIGHTS = {'pytorch': 0.4, 'keras': 0.35, 'color': 0.25}

# ==================== INITIAL SETUP (MOVED UP) ====================
app = Flask(__name__)
config = Config()

# Create directories
for directory in [config.IMAGES_DIR, config.LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging (THE FIX IS HERE)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'app.log')),
        logging.StreamHandler() # This will print logs to the console (and Render's log stream)
    ]
)
logger = logging.getLogger(__name__)

# ==================== LIBRARY AVAILABILITY CHECK (NOW SAFE TO USE LOGGER) ====================
if TENSORFLOW_AVAILABLE:
    if tf.config.list_physical_devices('GPU'):
        logger.info("GPU is available for TensorFlow.")
    else:
        logger.info("TensorFlow is using CPU.")
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
else:
    logger.warning("TensorFlow library not found.")

if not PYTORCH_AVAILABLE:
    logger.warning("PyTorch library not found.")
if not YOLO_AVAILABLE:
    logger.warning("Ultralytics YOLO library not found.")
if not CV2_AVAILABLE:
    logger.warning("OpenCV (cv2) library not found.")


# ==================== MEMORY MANAGEMENT ====================
def clear_memory():
    gc.collect()
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

# (The rest of the file remains the same as the previous version)
# ... (FallbackFruitClassifier, AdvancedColorAnalyzer, ModelManager, etc.) ...
# Just copy the rest of the classes and functions from the previous answer here.
# I will omit them for brevity, but they are required for the app to run.
# =======================================================================
# Start of the code that you should copy from the previous answer
# =======================================================================

# ==================== FALLBACK CLASSIFIER ====================
class FallbackFruitClassifier:
    """A simple fallback classifier if Keras model fails to load."""
    def __init__(self, classes):
        self.classes = classes
        logger.warning("Initializing FallbackFruitClassifier. Predictions will be basic.")

    def predict(self, image_array):
        mean_brightness = np.mean(image_array)
        # Return a dummy probability distribution
        if mean_brightness > 128: # Assume "fresh" for bright images
            # Higher probability for fresh classes
            preds = [0.25, 0.25, 0.25, 0.08, 0.08, 0.09]
        else: # Assume "rotten" for dark images
            # Higher probability for rotten classes
            preds = [0.08, 0.08, 0.09, 0.25, 0.25, 0.25]
        return [np.array(preds)]


# ==================== DOMINANT COLOR ANALYSIS ====================
class AdvancedColorAnalyzer:
    def analyze_image(self, image_path, bounding_box=None):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if bounding_box:
                x, y, w, h = bounding_box
                image = image[y:y+h, x:x+w]
            
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            hue_mean = np.mean(hsv_image[:, :, 0])
            saturation_mean = np.mean(hsv_image[:, :, 1])
            
            ripeness_state = "Không xác định"
            confidence = 0

            if 30 < hue_mean < 80 and saturation_mean > 50:
                ripeness_state = "Xanh"
                confidence = (saturation_mean / 255) * 100
            elif (0 < hue_mean < 30 or 160 < hue_mean < 180) and saturation_mean > 60:
                ripeness_state = "Chín"
                confidence = (saturation_mean / 255) * 100
            elif saturation_mean < 40:
                ripeness_state = "Quá chín / Hỏng"
                confidence = (1 - (saturation_mean / 255)) * 50
            
            color_vis_filename = f"{uuid.uuid4()}_color_analysis.jpg"
            color_vis_path = os.path.join(config.IMAGES_DIR, color_vis_filename)
            vis_img = Image.fromarray(image)
            draw = ImageDraw.Draw(vis_img)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            draw.text((10, 10), f"Color Ripeness: {ripeness_state}\nConfidence: {confidence:.1f}%", fill="white", font=font)
            vis_img.save(color_vis_path)

            return {
                'ripeness_state': ripeness_state,
                'confidence': confidence,
                'visualization_path': color_vis_filename
            }
        except Exception as e:
            logger.error(f"Error in AdvancedColorAnalyzer: {e}")
            return {
                'ripeness_state': 'Lỗi phân tích màu',
                'confidence': 0,
                'visualization_path': None
            }

# ==================== MODEL MANAGER ====================
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda:0" if PYTORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.models = {}
        self.keras_class_names = []
        self.load_all_models()

    def _load_keras_class_names(self):
        try:
            with open(config.CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
                class_indices = json.load(f)
            
            sorted_names = [""] * len(class_indices)
            for key, name in class_indices.items():
                sorted_names[int(key)] = name

            translation_map = {
                "freshapples": "Táo Tươi", "freshbanana": "Chuối Tươi", "freshoranges": "Cam Tươi",
                "rottenapples": "Táo Hỏng", "rottenbanana": "Chuối Hỏng", "rottenoranges": "Cam Hỏng"
            }
            self.keras_class_names = [translation_map.get(name, name) for name in sorted_names]
            logger.info(f"✅ Successfully loaded and mapped Keras class names.")
        except Exception as e:
            logger.error(f"❌ Could not load class indices from '{config.CLASS_INDICES_PATH}': {e}. Using fallback.")
            self.keras_class_names = config.FRESHNESS_CLASSES_FALLBACK

    def _load_keras_model(self):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow is not available. Using fallback classifier.")
            self.models['keras'] = FallbackFruitClassifier(self.keras_class_names)
            return

        for model_path in config.KERAS_MODEL_PATHS:
            if os.path.exists(model_path):
                try:
                    logger.info(f"Attempting to load Keras model from: {model_path}")
                    self.models['keras'] = load_model(model_path, compile=True)
                    logger.info(f"✅ Keras model loaded successfully from {model_path}.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {e}. Trying next path.")
        
        logger.error("❌ All Keras model paths failed. Using fallback classifier.")
        self.models['keras'] = FallbackFruitClassifier(self.keras_class_names)

    def _load_pytorch_model(self):
        if not PYTORCH_AVAILABLE:
            self.models['pytorch'] = None
            return

        if os.path.exists(config.RIPENESS_MODEL_PATH):
            try:
                logger.info("Loading PyTorch ripeness model...")
                model = models.mobilenet_v2(weights=None)
                num_ftrs = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(num_ftrs, len(config.RIPENESS_CLASSES))
                )
                model.load_state_dict(torch.load(config.RIPENESS_MODEL_PATH, map_location=self.device))
                self.models['pytorch'] = model.to(self.device).eval()
                logger.info("✅ PyTorch model loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Error loading PyTorch model: {e}")
                self.models['pytorch'] = None
        else:
            self.models['pytorch'] = None

    def _load_yolo_model(self):
        if not YOLO_AVAILABLE:
            self.models['yolo'] = None
            return
            
        if os.path.exists(config.DETECTOR_MODEL_PATH):
            try:
                logger.info("Loading YOLO detection model...")
                self.models['yolo'] = YOLO(config.DETECTOR_MODEL_PATH)
                logger.info("✅ YOLO model loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Error loading YOLO model: {e}")
                self.models['yolo'] = None
        else:
            self.models['yolo'] = None

    def load_all_models(self):
        logger.info("🚀 Initializing all models...")
        self._load_keras_class_names()
        self._load_keras_model()
        self._load_pytorch_model()
        self._load_yolo_model()
        logger.info("✅ Model initialization complete.")
        clear_memory()

model_manager = ModelManager()

# ==================== IMAGE PROCESSING ====================
class ImageProcessor:
    @staticmethod
    def validate_image(file_path_or_data, is_file=True):
        try:
            if is_file:
                if not os.path.exists(file_path_or_data): return False, "File không tồn tại"
                if os.path.getsize(file_path_or_data) > config.MAX_FILE_SIZE: return False, "File quá lớn"
                image = Image.open(file_path_or_data)
            else:
                image = Image.open(io.BytesIO(file_path_or_data))

            if image.format.lower() not in config.ALLOWED_EXT: return False, "Định dạng không hỗ trợ"
            if image.size[0] < 50 or image.size[1] < 50: return False, "Ảnh quá nhỏ"
            return True, "OK"
        except Exception as e:
            return False, f"Lỗi xử lý ảnh: {str(e)}"
    
    @staticmethod
    def process_image_from_link(link):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            with urllib.request.urlopen(link, timeout=15) as response:
                image_bytes = response.read()
            
            is_valid, message = ImageProcessor.validate_image(image_bytes, is_file=False)
            if not is_valid: return None, message

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            filename = f"{uuid.uuid4()}_url.jpg"
            saved_path = os.path.join(config.IMAGES_DIR, filename)
            image.save(saved_path, quality=95)
            return saved_path, None
        except Exception as e:
            logger.error(f"Error processing image from URL: {e}")
            return None, f"Lỗi xử lý ảnh từ URL: {str(e)}"

    @staticmethod
    def save_base64_image(data_url):
        try:
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            is_valid, message = ImageProcessor.validate_image(binary_data, is_file=False)
            if not is_valid: return None, message

            image = Image.open(io.BytesIO(binary_data)).convert("RGB")
            filename = f"{uuid.uuid4()}_camera.jpg"
            saved_path = os.path.join(config.IMAGES_DIR, filename)
            image.save(saved_path, quality=95)
            return saved_path, None
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return None, f"Lỗi xử lý ảnh từ camera: {str(e)}"

# ==================== FRUIT DETECTION & ANALYSIS ====================
class FruitAnalyzer:
    def __init__(self, model_manager_instance):
        self.models = model_manager_instance.models
        self.device = model_manager_instance.device

    def detect_fruit_with_yolo(self, image_path):
        if self.models.get('yolo') is None:
            return None
        try:
            results = self.models['yolo'](image_path, verbose=False)
            best_box = None
            max_confidence = 0
            for r in results:
                for box in r.boxes:
                    confidence = float(box.conf[0])
                    if confidence > max_confidence and confidence > config.CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0]
                        best_box = {'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 'confidence': confidence}
                        max_confidence = confidence
            clear_memory()
            return best_box
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return None

    def create_analysis_images(self, image_path, detection_result):
        try:
            original_image = Image.open(image_path).convert("RGB")
            images = {}
            original_filename = f"{uuid.uuid4()}_original.jpg"
            original_image.save(os.path.join(config.IMAGES_DIR, original_filename))
            images['original'] = original_filename

            bbox_image = original_image.copy()
            draw = ImageDraw.Draw(bbox_image)
            if detection_result and detection_result['bbox']:
                x, y, w, h = detection_result['bbox']
                draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
            bbox_filename = f"{uuid.uuid4()}_bbox.jpg"
            bbox_image.save(os.path.join(config.IMAGES_DIR, bbox_filename))
            images['bbox'] = bbox_filename

            if detection_result and detection_result['bbox']:
                x, y, w, h = detection_result['bbox']
                cropped_image = original_image.crop((x, y, x + w, y + h))
            else:
                width, height = original_image.size
                crop_size = min(width, height)
                left, top = (width - crop_size)//2, (height - crop_size)//2
                cropped_image = original_image.crop((left, top, left + crop_size, top + crop_size))
            cropped_filename = f"{uuid.uuid4()}_cropped.jpg"
            cropped_path = os.path.join(config.IMAGES_DIR, cropped_filename)
            cropped_image.save(cropped_path)
            images['cropped'] = cropped_filename
            images['color'] = ''

            return images
        except Exception as e:
            logger.error(f"Error creating analysis images: {e}")
            return None

    def predict_freshness_keras(self, image_path):
        if self.models.get('keras') is None:
            return {'classes': ["Lỗi model"], 'probabilities': [0], 'confidence': 0}
        try:
            img = load_img(image_path, target_size=config.TARGET_SIZE)
            img_array = img_to_array(img)
            
            # Use the correct predict method based on the model type
            if isinstance(self.models['keras'], FallbackFruitClassifier):
                 predictions = self.models['keras'].predict(img_array)[0]
            else:
                 img_array_reshaped = img_array.reshape(1, *config.TARGET_SIZE, 3).astype('float32') / 255.0
                 predictions = self.models['keras'].predict(img_array_reshaped, verbose=0)[0]

            top_indices = predictions.argsort()[::-1][:3]
            results = {
                'classes': [model_manager.keras_class_names[i] for i in top_indices],
                'probabilities': [float(predictions[i] * 100) for i in top_indices],
                'confidence': float(predictions[top_indices[0]] * 100)
            }
            clear_memory()
            return results
        except Exception as e:
            logger.error(f"Keras prediction error: {e}")
            return {'classes': ["Lỗi dự đoán"], 'probabilities': [0], 'confidence': 0}

    def predict_ripeness_pytorch(self, image_path):
        if self.models.get('pytorch') is None:
            return {'class': "Không có model", 'confidence': 0}
        try:
            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.models['pytorch'](input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            results = {
                'class': config.RIPENESS_CLASSES[predicted_idx.item()],
                'confidence': float(confidence.item() * 100)
            }
            clear_memory()
            return results
        except Exception as e:
            logger.error(f"PyTorch prediction error: {e}")
            return {'class': "Lỗi dự đoán", 'confidence': 0}
    
    def analyze_color_ripeness(self, image_path, detection_result):
        if not CV2_AVAILABLE:
            return {'ripeness': 'Không thể phân tích', 'confidence': 0, 'visualization_path': None}
        try:
            color_analyzer = AdvancedColorAnalyzer()
            bbox = detection_result['bbox'] if detection_result else None
            analysis_result = color_analyzer.analyze_image(image_path, bounding_box=bbox)
            return {
                'ripeness': analysis_result['ripeness_state'],
                'confidence': analysis_result['confidence'],
                'visualization_path': analysis_result['visualization_path']
            }
        except Exception as e:
            logger.error(f"Advanced Color analysis error: {e}")
            return {'ripeness': 'Lỗi phân tích', 'confidence': 0, 'visualization_path': None}

# ==================== RESULT AGGREGATOR ====================
class ResultAggregator:
    @staticmethod
    def extract_fruit_type(pytorch_result, keras_result):
        pytorch_class = pytorch_result.get('class', '').lower()
        keras_class = keras_result.get('classes', [''])[0].lower()
        fruit_mapping = {'apple': 'Táo', 'banana': 'Chuối', 'orange': 'Cam'}
        for eng, viet in fruit_mapping.items():
            if eng in pytorch_class: return viet
        for eng, viet in fruit_mapping.items():
            if viet.lower() in keras_class: return viet
        return "Không xác định"

    @staticmethod
    def extract_ripeness(pytorch_result, color_result):
        pytorch_class = pytorch_result.get('class', '').lower()
        color_ripeness = color_result.get('ripeness', 'không xác định').lower()
        pytorch_conf = pytorch_result.get('confidence', 0)
        
        if pytorch_conf > 75:
            if 'unripe' in pytorch_class: return "Xanh"
            if 'ripe' in pytorch_class: return "Chín"
        
        if 'xanh' in color_ripeness: return "Xanh"
        if 'chín' in color_ripeness: return "Chín"
        return "Không xác định"

    @staticmethod
    def extract_freshness(keras_result):
        top_class = keras_result.get('classes', [''])[0].lower()
        if 'tươi' in top_class: return 'Tươi'
        if 'hỏng' in top_class: return 'Hỏng'
        return "Không xác định"

    @staticmethod
    def calculate_overall_confidence(pytorch_result, keras_result, color_result):
        weights = config.ENSEMBLE_WEIGHTS
        return round(
            pytorch_result.get('confidence', 0) * weights['pytorch'] +
            keras_result.get('confidence', 0) * weights['keras'] +
            color_result.get('confidence', 0) * weights['color'], 2
        )

    @staticmethod
    def generate_recommendation(ripeness, freshness, confidence):
        if confidence < 40: return "Độ tin cậy thấp. Vui lòng thử với ảnh rõ nét hơn."
        if freshness == "Hỏng": return "⚠️ Trái cây đã hỏng, không nên sử dụng."
        if freshness == "Tươi":
            if ripeness == "Chín": return "✅ Trái cây tươi và chín, sẵn sàng để ăn."
            if ripeness == "Xanh": return "🕐 Trái cây tươi nhưng chưa chín, cần để thêm thời gian."
            return "✅ Trái cây tươi, chất lượng tốt."
        return "ℹ️ Cần kiểm tra thêm để đánh giá chính xác."

    @staticmethod
    def calculate_quality_score(ripeness, freshness, confidence):
        base_score = confidence
        if freshness == "Hỏng": base_score *= 0.2
        elif freshness == "Tươi": base_score *= 1.0
        else: base_score *= 0.6
        if ripeness == "Chín": base_score *= 1.0
        elif ripeness == "Xanh": base_score *= 0.8
        else: base_score *= 0.7
        return round(min(100, max(0, base_score)), 1)
    
    @staticmethod
    def generate_final_result(pytorch_result, keras_result, color_result):
        try:
            fruit_type = ResultAggregator.extract_fruit_type(pytorch_result, keras_result)
            ripeness = ResultAggregator.extract_ripeness(pytorch_result, color_result)
            freshness = ResultAggregator.extract_freshness(keras_result)
            overall_confidence = ResultAggregator.calculate_overall_confidence(pytorch_result, keras_result, color_result)
            
            final_description = f"{fruit_type} {ripeness} {freshness}".replace("Không xác định", "").strip().replace("  ", " ")
            if not final_description: final_description = "Không thể xác định"

            return {
                'description': final_description,
                'fruit_type': fruit_type, 'ripeness': ripeness, 'freshness': freshness,
                'overall_confidence': overall_confidence,
                'recommendation': ResultAggregator.generate_recommendation(ripeness, freshness, overall_confidence),
                'quality_score': ResultAggregator.calculate_quality_score(ripeness, freshness, overall_confidence)
            }
        except Exception as e:
            logger.error(f"Error generating final result: {e}")
            return {'description': "Lỗi xử lý kết quả", 'overall_confidence': 0}


fruit_analyzer = FruitAnalyzer(model_manager)

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", "timestamp": datetime.utcnow().isoformat(),
        "libraries": {"tensorflow": TENSORFLOW_AVAILABLE, "pytorch": PYTORCH_AVAILABLE, "yolo": YOLO_AVAILABLE, "opencv": CV2_AVAILABLE},
        "models_loaded": {
            "keras": "Fallback" if isinstance(model_manager.models.get('keras'), FallbackFruitClassifier) else (model_manager.models.get('keras') is not None),
            "pytorch": model_manager.models.get('pytorch') is not None,
            "yolo": model_manager.models.get('yolo') is not None,
        }
    })

@app.route('/success', methods=['POST'])
def success():
    error = ''
    img_path = None
    try:
        if 'link' in request.form and request.form.get('link'):
            img_path, error = ImageProcessor.process_image_from_link(request.form.get('link').strip())
        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXT:
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                img_path = os.path.join(config.IMAGES_DIR, filename)
                file.save(img_path)
            else:
                error = "Định dạng ảnh không hợp lệ."
        elif 'camera_image' in request.form and request.form.get('camera_image'):
            img_path, error = ImageProcessor.save_base64_image(request.form.get('camera_image'))
        else:
            error = "Vui lòng tải ảnh, nhập liên kết hoặc chụp ảnh."

        if img_path and not error:
            logger.info(f"Starting analysis for image: {os.path.basename(img_path)}")
            
            detection_result = fruit_analyzer.detect_fruit_with_yolo(img_path)
            analysis_images = fruit_analyzer.create_analysis_images(img_path, detection_result)
            if not analysis_images:
                error = "Lỗi tạo ảnh phân tích."
                return render_template("index.html", error=error)
            
            cropped_path = os.path.join(config.IMAGES_DIR, analysis_images['cropped'])
            
            keras_result = fruit_analyzer.predict_freshness_keras(cropped_path)
            pytorch_result = fruit_analyzer.predict_ripeness_pytorch(cropped_path)
            color_result = fruit_analyzer.analyze_color_ripeness(img_path, detection_result)

            if color_result.get('visualization_path'):
                analysis_images['color'] = color_result['visualization_path']
            
            final_result = ResultAggregator.generate_final_result(pytorch_result, keras_result, color_result)

            predictions_for_template = {
                "freshness_details": keras_result,
                "color_ripeness": color_result.get('ripeness'),
                "color_confidence": color_result.get('confidence'),
                "pytorch_prediction": pytorch_result.get('class'),
                "pytorch_confidence": pytorch_result.get('confidence'),
                "final_result": final_result,
                "detection_confidence": detection_result['confidence'] if detection_result else 0,
                "yolo_detected": detection_result is not None
            }
            
            return render_template("success.html", images=analysis_images, predictions=predictions_for_template)
    except Exception as e:
        logger.error(f"FATAL error in /success route: {e}", exc_info=True)
        error = f"Đã xảy ra lỗi hệ thống nghiêm trọng: {str(e)}"
    
    return render_template("index.html", error=error)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error (500): {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("Starting Synthesized Fruit Analysis Application...")
    port = int(os.environ.get("PORT", 5000))
    # For production, use a production-ready server like gunicorn instead of app.run()
    # But for Render/local testing, this is fine.
    app.run(debug=False, host='0.0.0.0', port=port)
