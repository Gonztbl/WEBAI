import os, io, uuid, urllib.request, json, logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from dominant_color import AdvancedColorAnalyzer, ColorAnalysisResult, RipenessState, color_of_image, name_main_color
from ultralytics import YOLO
import cv2
import base64


# ==================== CONFIGURATION ====================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'fruit_class_indices.json')
    # Model paths
    CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_state_classifier.keras')
    DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'yolo11n.pt')
    RIPENESS_MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_ripeness_model_pytorch.pth')

    # Classes
    FRESHNESS_CLASSES = ['Táo Tươi', 'Chuối Tươi', 'Cam Tươi', 'Táo Hỏng', 'Chuối Hỏng', 'Cam Hỏng']
    RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
    FRUIT_TYPES = ['Apple', 'Banana', 'Orange']

    # File settings
    ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif', 'webp'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # Processing settings
    TARGET_SIZE = (224, 224)
    CONFIDENCE_THRESHOLD = 0.6
    ENSEMBLE_WEIGHTS = {
        'pytorch': 0.4,
        'keras': 0.35,
        'color': 0.25
    }


# ==================== SETUP ====================
app = Flask(__name__)
config = Config()

# Create directories
for directory in [config.IMAGES_DIR, config.LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== MODEL MANAGER ====================
# ==================== MODEL MANAGER ====================
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = {}
        # Mới: Khởi tạo danh sách nhãn Keras
        self.keras_class_names_vietnamese = []
        self.load_models()

    # THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM NÀY
    def _load_keras_class_names(self):
        """Tải và xử lý tên các lớp cho mô hình Keras từ file JSON."""
        try:
            logger.info("Loading Keras class indices from JSON...")
            # Đảm bảo bạn đang dùng đúng tên file đã cấu hình trong class Config
            with open(config.CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
                # class_indices sẽ có dạng: {"0": "freshapples", "1": "freshbanana", ...}
                class_indices_from_json = json.load(f)

            # Tạo một list rỗng với kích thước bằng số lượng lớp
            sorted_class_names = [""] * len(class_indices_from_json)

            # Lặp qua dictionary từ file JSON
            # key_str là chỉ số dưới dạng chuỗi (ví dụ: "0")
            # class_name là tên lớp (ví dụ: "freshapples")
            for key_str, class_name in class_indices_from_json.items():
                # Chuyển key từ chuỗi sang số nguyên để làm chỉ số cho list
                index = int(key_str)
                # Đặt tên lớp vào đúng vị trí trong list
                # Ví dụ: sorted_class_names[0] = "freshapples"
                sorted_class_names[index] = class_name

            # Tạo một map để dịch sang tiếng Việt
            translation_map = {
                "freshapples": "Táo Tươi", "freshbanana": "Chuối Tươi", "freshoranges": "Cam Tươi",
                "rottenapples": "Táo Hỏng", "rottenbanana": "Chuối Hỏng", "rottenoranges": "Cam Hỏng"
            }

            # Dùng map để tạo danh sách nhãn tiếng Việt cuối cùng
            self.keras_class_names_vietnamese = [
                translation_map.get(name, name) for name in sorted_class_names
            ]

            logger.info("Keras class names loaded and mapped successfully.")
            logger.info(f"Loaded labels: {self.keras_class_names_vietnamese}")

        except FileNotFoundError:
            # Nhớ sửa lại tên biến config cho đúng như đã thảo luận ở câu trả lời trước
            logger.error(f"FATAL: Keras class indices file not found at '{config.CLASS_INDICES_PATH}'")
            logger.warning("Falling back to hardcoded list. THIS IS UNRELIABLE and may cause wrong predictions!")
            # Đảm bảo biến fallback tồn tại trong Config
            self.keras_class_names_vietnamese = config.FRESHNESS_CLASSES_FALLBACK
        except Exception as e:
            logger.error(f"Error loading Keras class indices: {e}")
            raise
    def load_models(self):
        try:
            # Load Keras model
            logger.info("Loading Keras freshness model...")
            self.models['keras'] = load_model(config.CLASSIFIER_MODEL_PATH)

            # Mới: Tải nhãn cho mô hình Keras
            self._load_keras_class_names()

            # Load YOLO model
            logger.info("Loading YOLO detection model...")
            self.models['yolo'] = YOLO(config.DETECTOR_MODEL_PATH)

            # Load PyTorch model
            logger.info("Loading PyTorch ripeness model...")
            pytorch_model = models.mobilenet_v2(weights=None)
            num_ftrs = pytorch_model.classifier[1].in_features
            pytorch_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_ftrs, len(config.RIPENESS_CLASSES))
            )
            pytorch_model.load_state_dict(torch.load(config.RIPENESS_MODEL_PATH, map_location=self.device))
            self.models['pytorch'] = pytorch_model.to(self.device).eval()

            logger.info("All models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise


# Khởi tạo ngay sau khi định nghĩa class
model_manager = ModelManager()
# ==================== IMAGE PROCESSING ====================
class ImageProcessor:
    @staticmethod
    def validate_image(file_path_or_data, is_file=True):
        """Validate image file or data"""
        try:
            if is_file:
                if not os.path.exists(file_path_or_data):
                    return False, "File không tồn tại"

                file_size = os.path.getsize(file_path_or_data)
                if file_size > config.MAX_FILE_SIZE:
                    return False, f"File quá lớn (tối đa {config.MAX_FILE_SIZE // 1024 // 1024}MB)"

                image = Image.open(file_path_or_data)
            else:
                image = Image.open(io.BytesIO(file_path_or_data))

            # Check image format
            if image.format.lower() not in ['jpeg', 'jpg', 'png', 'webp']:
                return False, "Định dạng ảnh không được hỗ trợ"

            # Check image dimensions
            if image.size[0] < 50 or image.size[1] < 50:
                return False, "Ảnh quá nhỏ (tối thiểu 50x50px)"

            return True, "OK"

        except Exception as e:
            return False, f"Lỗi xử lý ảnh: {str(e)}"

    @staticmethod
    def process_image_from_link(link):
        """Download and process image from URL"""
        try:
            logger.info(f"Processing image from URL: {link}")

            # Validate URL
            if not link.startswith(('http://', 'https://')):
                return None, "URL không hợp lệ"

            # Download image
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
            urllib.request.install_opener(opener)

            with urllib.request.urlopen(link, timeout=15) as response:
                if response.getcode() != 200:
                    return None, "Không thể tải ảnh từ URL"

                image_bytes = response.read()

            # Validate image data
            is_valid, message = ImageProcessor.validate_image(image_bytes, is_file=False)
            if not is_valid:
                return None, message

            # Save image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            filename = f"{uuid.uuid4()}_url.jpg"
            saved_path = os.path.join(config.IMAGES_DIR, filename)
            image.save(saved_path, quality=95)

            logger.info(f"Image saved successfully: {filename}")
            return saved_path, None

        except urllib.error.URLError as e:
            return None, f"Lỗi kết nối: {str(e)}"
        except Exception as e:
            logger.error(f"Error processing image from URL: {e}")
            return None, f"Lỗi xử lý ảnh từ URL: {str(e)}"

    @staticmethod
    def save_base64_image(data_url):
        """Save base64 encoded image"""
        try:
            logger.info("Processing base64 image from camera")

            if not data_url.startswith('data:image'):
                return None, "Dữ liệu ảnh không hợp lệ"

            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)

            # Validate image data
            is_valid, message = ImageProcessor.validate_image(binary_data, is_file=False)
            if not is_valid:
                return None, message

            # Save image
            image = Image.open(io.BytesIO(binary_data)).convert("RGB")
            filename = f"{uuid.uuid4()}_camera.jpg"
            saved_path = os.path.join(config.IMAGES_DIR, filename)
            image.save(saved_path, quality=95)

            logger.info(f"Camera image saved: {filename}")
            return saved_path, None

        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return None, f"Lỗi xử lý ảnh từ camera: {str(e)}"


# ==================== FRUIT DETECTION & ANALYSIS ====================
class FruitAnalyzer:
    def __init__(self, model_manager):
        self.models = model_manager.models
        self.device = model_manager.device

    def detect_fruit_with_yolo(self, image_path):
        """Detect fruit using YOLO and return best bounding box"""
        try:
            logger.info("Running YOLO fruit detection...")

            results = self.models['yolo'](image_path, verbose=False)
            best_box = None
            max_confidence = 0

            for r in results:
                for box in r.boxes:
                    confidence = float(box.conf[0])
                    if confidence > max_confidence and confidence > config.CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0]
                        best_box = {
                            'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            'confidence': confidence,
                            'class_id': int(box.cls[0]) if box.cls is not None else 0
                        }
                        max_confidence = confidence

            logger.info(f"YOLO detection result: {best_box}")
            return best_box

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return None

    def create_analysis_images(self, image_path, detection_result):
        """Create analysis images: original, bbox, cropped."""  # Removed color analysis from here
        try:
            original_image = Image.open(image_path).convert("RGB")
            images = {}

            # 1. Original image
            original_filename = f"{uuid.uuid4()}_original.jpg"
            original_path = os.path.join(config.IMAGES_DIR, original_filename)
            original_image.save(original_path, quality=95)
            images['original'] = original_filename

            # 2. Bounding box image
            bbox_image = original_image.copy()
            draw = ImageDraw.Draw(bbox_image)

            if detection_result and detection_result['bbox']:
                x, y, w, h = detection_result['bbox']
                confidence = detection_result['confidence']
                draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                label = f"Fruit: {confidence:.2f}"
                draw.text((x, y - 25), label, fill="red", font=font)

            bbox_filename = f"{uuid.uuid4()}_bbox.jpg"
            bbox_path = os.path.join(config.IMAGES_DIR, bbox_filename)
            bbox_image.save(bbox_path, quality=95)
            images['bbox'] = bbox_filename

            # 3. Cropped image for classification
            if detection_result and detection_result['bbox']:
                x, y, w, h = detection_result['bbox']
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(original_image.width - x, w + 2 * padding)
                h = min(original_image.height - y, h + 2 * padding)
                cropped_image = original_image.crop((x, y, x + w, y + h))
            else:
                width, height = original_image.size
                crop_size = min(width, height)
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
                cropped_image = original_image.crop((left, top, left + crop_size, top + crop_size))

            cropped_filename = f"{uuid.uuid4()}_cropped.jpg"
            cropped_path = os.path.join(config.IMAGES_DIR, cropped_filename)
            cropped_image.save(cropped_path, quality=95)
            images['cropped'] = cropped_filename

            # Color analysis image will be generated and added later
            images['color'] = ''  # Placeholder

            logger.info(f"Created base analysis images: {list(images.keys())}")
            return images

        except Exception as e:
            logger.error(f"Error creating analysis images: {e}")
            return None
    def predict_freshness_keras(self, image_path):
        """Predict freshness using Keras model"""
        try:
            logger.info("Running Keras freshness prediction...")

            img = load_img(image_path, target_size=config.TARGET_SIZE)
            img_array = img_to_array(img).reshape(1, *config.TARGET_SIZE, 3).astype('float32') / 255.0

            predictions = self.models['keras'].predict(img_array, verbose=0)[0]

            # Get top 3 predictions
            top_indices = predictions.argsort()[::-1][:3]
            results = {
                'classes': [model_manager.keras_class_names_vietnamese[i] for i in top_indices],
                'probabilities': [float(predictions[i] * 100) for i in top_indices],
                'confidence': float(predictions[top_indices[0]] * 100)
            }

            logger.info(f"Keras prediction: {results['classes'][0]} ({results['confidence']:.2f}%)")
            return results

        except Exception as e:
            logger.error(f"Keras prediction error: {e}")
            return {
                'classes': ["Không xác định", "Không xác định", "Không xác định"],
                'probabilities': [0, 0, 0],
                'confidence': 0
            }

    def predict_ripeness_pytorch(self, image_path):
        """Predict ripeness using PyTorch model"""
        try:
            logger.info("Running PyTorch ripeness prediction...")

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.models['pytorch'](input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

            results = {
                'class': config.RIPENESS_CLASSES[predicted_idx.item()],
                'confidence': float(confidence.item() * 100),
                'all_probabilities': [float(p * 100) for p in probabilities.cpu().numpy()]
            }

            logger.info(f"PyTorch prediction: {results['class']} ({results['confidence']:.2f}%)")
            return results

        except Exception as e:
            logger.error(f"PyTorch prediction error: {e}")
            return {
                'class': "Không xác định",
                'confidence': 0,
                'all_probabilities': [0] * len(config.RIPENESS_CLASSES)
            }

    def analyze_color_ripeness(self, image_path, detection_result):
        """Analyze color for ripeness using the advanced analyzer."""
        try:
            logger.info("Running Advanced Color Analysis for ripeness...")
            color_analyzer = AdvancedColorAnalyzer()
            bbox = detection_result['bbox'] if detection_result and 'bbox' in detection_result else None

            # This is where the advanced analysis happens
            analysis_result = color_analyzer.analyze_image(image_path, bounding_box=bbox)

            results = {
                'ripeness': analysis_result.ripeness_state.value,
                'confidence': analysis_result.confidence,
                'visualization_path': analysis_result.visualization_path
            }

            logger.info(f"Advanced Color Analysis result: {results['ripeness']} ({results['confidence']:.2f}%)")
            return results

        except Exception as e:
            logger.error(f"Advanced Color analysis error: {e}")
            return {
                'ripeness': "Không xác định",
                'confidence': 0,
                'visualization_path': ''
            }

# ==================== RESULT AGGREGATOR ====================
class ResultAggregator:
    @staticmethod
    def extract_fruit_type(pytorch_result, keras_result):
        """Extract fruit type from predictions"""
        # Priority: PyTorch > Keras
        pytorch_class = pytorch_result.get('class', '').lower()
        keras_class = keras_result.get('classes', [''])[0].lower()

        fruit_mapping = {
            'apple': 'Táo',
            'banana': 'Chuối',
            'orange': 'Cam'
        }

        # Check PyTorch result first
        for eng, viet in fruit_mapping.items():
            if eng in pytorch_class:
                return viet

        # Check Keras result
        for eng, viet in fruit_mapping.items():
            if viet.lower() in keras_class:
                return viet

        return "Không xác định"

    @staticmethod
    @staticmethod
    @staticmethod
    def extract_ripeness(pytorch_result, color_result):
        """Extracts ripeness using a more robust, hierarchical logic."""
        pytorch_class = pytorch_result.get('class', '').lower()
        color_ripeness_str = color_result.get('ripeness', 'KHÔNG XÁC ĐỊNH').upper()
        pytorch_conf = pytorch_result.get('confidence', 0)
        color_conf = color_result.get('confidence', 0)

        # 1. Xác định kết quả từ PyTorch (nguồn đáng tin cậy nhất)
        pytorch_verdict = "Không xác định"
        if 'unripe' in pytorch_class:
            pytorch_verdict = "Xanh"
        elif 'ripe' in pytorch_class:
            pytorch_verdict = "Chín"

        # 2. Xác định kết quả từ phân tích màu
        color_verdict = "Không xác định"
        if color_ripeness_str == 'XANH':
            color_verdict = "Xanh"
        elif color_ripeness_str == 'CHÍN':
            color_verdict = "Chín"

        # === QUY TẮC QUYẾT ĐỊNH (LOGIC MỚI QUAN TRỌNG NHẤT) ===

        # Quy tắc 1: Nếu PyTorch cực kỳ chắc chắn (> 90%), quyết định của nó là cuối cùng.
        # Đây là quy tắc quan trọng nhất để sửa lỗi hiện tại.
        if pytorch_conf > 90:
            return pytorch_verdict

        # Quy tắc 2: Nếu mô hình màu không chắc chắn, bỏ qua nó và dùng PyTorch.
        if color_verdict == "Không xác định" or color_conf < 40:
            return pytorch_verdict

        # Quy tắc 3: Nếu cả hai mô hình đồng ý, tuyệt vời.
        if pytorch_verdict == color_verdict:
            return pytorch_verdict

        # Quy tắc 4 (Bất đồng): Tin vào mô hình có độ tin cậy cao hơn.
        if pytorch_conf > color_conf:
            return pytorch_verdict
        else:
            return color_verdict

        # Dự phòng cuối cùng: Luôn tin vào PyTorch nếu không có quy tắc nào ở trên được áp dụng.
        return pytorch_verdict

    @staticmethod
    def extract_freshness(keras_result):
        """Extract freshness from Keras prediction"""
        if not keras_result.get('classes'):
            return "Không xác định"

        top_class = keras_result['classes'][0].lower()

        if 'tươi' in top_class:
            return 'Tươi'
        elif 'hỏng' in top_class:
            return 'Hỏng'

        return "Không xác định"

    @staticmethod
    def calculate_overall_confidence(pytorch_result, keras_result, color_result):
        """Calculate weighted overall confidence"""
        pytorch_conf = pytorch_result.get('confidence', 0)
        keras_conf = keras_result.get('confidence', 0)
        color_conf = color_result.get('confidence', 0)

        weights = config.ENSEMBLE_WEIGHTS
        overall_confidence = (
                pytorch_conf * weights['pytorch'] +
                keras_conf * weights['keras'] +
                color_conf * weights['color']
        )

        return round(overall_confidence, 2)

    @staticmethod
    def generate_final_result(pytorch_result, keras_result, color_result):
        """Generate comprehensive final result"""
        try:
            # Extract components
            fruit_type = ResultAggregator.extract_fruit_type(pytorch_result, keras_result)
            ripeness = ResultAggregator.extract_ripeness(pytorch_result, color_result)
            freshness = ResultAggregator.extract_freshness(keras_result)

            # Calculate overall confidence
            overall_confidence = ResultAggregator.calculate_overall_confidence(
                pytorch_result, keras_result, color_result
            )

            # Generate final description
            if fruit_type != "Không xác định":
                if ripeness != "Không xác định" and freshness != "Không xác định":
                    final_description = f"{fruit_type} {ripeness} {freshness}"
                elif ripeness != "Không xác định":
                    final_description = f"{fruit_type} {ripeness}"
                elif freshness != "Không xác định":
                    final_description = f"{fruit_type} {freshness}"
                else:
                    final_description = fruit_type
            else:
                final_description = "Không thể xác định"

            # Generate recommendation
            recommendation = ResultAggregator.generate_recommendation(
                fruit_type, ripeness, freshness, overall_confidence
            )

            final_result = {
                'description': final_description,
                'fruit_type': fruit_type,
                'ripeness': ripeness,
                'freshness': freshness,
                'overall_confidence': overall_confidence,
                'recommendation': recommendation,
                'quality_score': ResultAggregator.calculate_quality_score(ripeness, freshness, overall_confidence)
            }

            logger.info(f"Final result: {final_description} (Confidence: {overall_confidence}%)")
            return final_result

        except Exception as e:
            logger.error(f"Error generating final result: {e}")
            return {
                'description': "Lỗi xử lý kết quả",
                'fruit_type': "Không xác định",
                'ripeness': "Không xác định",
                'freshness': "Không xác định",
                'overall_confidence': 0,
                'recommendation': "Vui lòng thử lại với ảnh khác",
                'quality_score': 0
            }

    @staticmethod
    def generate_recommendation(fruit_type, ripeness, freshness, confidence):
        """Generate recommendation based on analysis"""
        if confidence < 50:
            return "Độ tin cậy thấp. Vui lòng thử với ảnh rõ nét hơn."

        if freshness == "Hỏng":
            return "⚠️ Trái cây đã hỏng, không nên sử dụng."
        elif freshness == "Tươi":
            if ripeness == "Chín":
                return "✅ Trái cây tươi và chín, sẵn sàng để ăn."
            elif ripeness == "Xanh":
                return "🕐 Trái cây tươi nhưng chưa chín, cần để thêm thời gian."
            else:
                return "✅ Trái cây tươi, chất lượng tốt."
        else:
            return "ℹ️ Cần kiểm tra thêm để đánh giá chính xác."

    @staticmethod
    def calculate_quality_score(ripeness, freshness, confidence):
        """Calculate quality score from 0-100"""
        base_score = confidence

        if freshness == "Hỏng":
            base_score *= 0.2
        elif freshness == "Tươi":
            base_score *= 1.0
        else:
            base_score *= 0.6

        if ripeness == "Chín":
            base_score *= 1.0
        elif ripeness == "Xanh":
            base_score *= 0.8
        else:
            base_score *= 0.7

        return round(min(100, max(0, base_score)), 1)


# Initialize analyzer
fruit_analyzer = FruitAnalyzer(model_manager)


# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_cameras')
def get_cameras():
    """API endpoint to get available cameras"""
    try:
        # This is a mock response - in real implementation, you'd detect available cameras
        cameras = [
            {'id': 0, 'name': 'Camera trước', 'facing': 'user'},
            {'id': 1, 'name': 'Camera sau', 'facing': 'environment'}
        ]
        return jsonify({'success': True, 'cameras': cameras})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/success', methods=['POST'])
def success():
    error = ''
    img_path = None

    try:
        # Process different input types
        if 'link' in request.form and request.form.get('link'):
            img_path, error = ImageProcessor.process_image_from_link(request.form.get('link').strip())
        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXT:
                # Validate file
                file_data = file.read()
                is_valid, validation_message = ImageProcessor.validate_image(file_data, is_file=False)
                if not is_valid:
                    error = validation_message
                else:
                    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                    img_path = os.path.join(config.IMAGES_DIR, filename)

                    # Reset file pointer and save
                    file.seek(0)
                    file.save(img_path)
            else:
                error = "Định dạng ảnh không hợp lệ. Chỉ hỗ trợ JPG, PNG, JPEG, WEBP."
        elif 'camera_image' in request.form and request.form.get('camera_image'):
            img_path, error = ImageProcessor.save_base64_image(request.form.get('camera_image'))
        else:
            error = "Vui lòng tải ảnh, nhập liên kết hoặc chụp ảnh từ camera."

        if img_path and not error:
            logger.info(f"Starting analysis for image: {os.path.basename(img_path)}")

            # Step 1: Detect fruit with YOLO
            detection_result = fruit_analyzer.detect_fruit_with_yolo(img_path)

            # Step 2: Create analysis images (without color analysis part)
            # We will run color analysis separately to get the full result object
            analysis_images = fruit_analyzer.create_analysis_images(img_path, detection_result)
            if not analysis_images:
                error = "Lỗi tạo ảnh phân tích"
                return render_template("index.html", error=error)

            # Step 3: Run predictions and analyses
            cropped_path = os.path.join(config.IMAGES_DIR, analysis_images['cropped'])

            # Keras freshness prediction
            keras_result = fruit_analyzer.predict_freshness_keras(cropped_path)

            # PyTorch ripeness prediction
            pytorch_result = fruit_analyzer.predict_ripeness_pytorch(cropped_path)

            # Advanced Color analysis
            color_result = fruit_analyzer.analyze_color_ripeness(img_path, detection_result)

            # Update the analysis_images dict with the new visualization from the advanced analyzer
            if color_result.get('visualization_path'):
                analysis_images['color'] = color_result['visualization_path']

            # Step 4: Generate final aggregated result
            final_result = ResultAggregator.generate_final_result(
                pytorch_result, keras_result, color_result
            )

            # Prepare results for template
            predictions = {
                # Individual model results
                "freshness_class1": keras_result['classes'][0],
                "freshness_prob1": keras_result['probabilities'][0],
                "freshness_class2": keras_result['classes'][1],
                "freshness_prob2": keras_result['probabilities'][1],
                "freshness_class3": keras_result['classes'][2],
                "freshness_prob3": keras_result['probabilities'][2],
                "color_ripeness": color_result['ripeness'],
                "color_confidence": color_result['confidence'],
                "pytorch_prediction": pytorch_result['class'],
                "pytorch_confidence": pytorch_result['confidence'],

                # Final aggregated result
                "final_result": final_result,

                # Detection info
                "detection_confidence": detection_result['confidence'] if detection_result else 0,
                "yolo_detected": detection_result is not None
            }

            # Log successful analysis
            logger.info(f"Analysis completed successfully: {final_result['description']}")

            return render_template("success.html",
                                   images=analysis_images,
                                   predictions=predictions,
                                   detection_result=detection_result)

    except Exception as e:
        logger.error(f"Error in /success route: {e}")
        error = f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"

    return render_template("index.html", error=error)


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    logger.info("Starting Fruit Analysis Application...")
    app.run(debug=False, host='0.0.0.0', port=5000)
