# app.py (đã nâng cấp giao diện và logic vẽ bbox)
import os, io, uuid, urllib.request
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

# Cấu hình
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Cho phép định dạng ảnh
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

# Đường dẫn model
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fruit_state_classifier.keras')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'yolo11n.pt')
RIPENESS_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fruit_ripeness_model_pytorch.pth')

# Các lớp
FRESHNESS_CLASSES = ['Táo Tươi', 'Chuối Tươi', 'Cam Tươi', 'Táo Hỏng', 'Chuối Hỏng', 'Cam Hỏng']
RIPENESS_CLASSES = ['Apple Ripe', 'Apple Unripe', 'Banana Ripe', 'Banana Unripe', 'Orange Ripe', 'Orange Unripe']
NUM_PYTORCH_CLASSES = len(RIPENESS_CLASSES)

# Load mô hình
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier_model = load_model(CLASSIFIER_MODEL_PATH)
detector_model = YOLO(DETECTOR_MODEL_PATH)

ripeness_model = models.mobilenet_v2(weights=None)
num_ftrs = ripeness_model.classifier[1].in_features
ripeness_model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, NUM_PYTORCH_CLASSES))
ripeness_model.load_state_dict(torch.load(RIPENESS_MODEL_PATH, map_location=device))
ripeness_model = ripeness_model.to(device).eval()


# ==================== HÀM HỖ TRỢ ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def process_image_from_link(link):
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        with urllib.request.urlopen(link, timeout=10) as response:
            image_bytes = response.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        filename = str(uuid.uuid4()) + ".jpg"
        saved_path = os.path.join(IMAGES_DIR, filename)
        image.save(saved_path)
        return saved_path, None
    except:
        return None, "Lỗi khi xử lý liên kết ảnh."

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
    except:
        return None

def draw_yolo_bounding_box(image_path, bounding_box):
    image = Image.open(image_path).convert("RGB")
    if bounding_box:
        draw = ImageDraw.Draw(image)
        x, y, w, h = bounding_box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
    saved_name = str(uuid.uuid4()) + "_yolo.jpg"
    save_path = os.path.join(IMAGES_DIR, saved_name)
    image.save(save_path)
    return saved_name

def predict_fruit_freshness(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img).reshape(1, 224, 224, 3).astype('float32') / 255.0
    preds = classifier_model.predict(img, verbose=0)[0]
    top = preds.argsort()[::-1][:3]
    return [FRESHNESS_CLASSES[i] for i in top], [(preds[i]*100).round(2) for i in top]

def predict_ripeness_pytorch(image_path):
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


# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods=['POST'])
def success():
    error = ''
    img_path = None
    try:
        if 'link' in request.form and request.form.get('link'):
            img_path, error = process_image_from_link(request.form.get('link').strip())
        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if allowed_file(file.filename):
                filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
                img_path = os.path.join(IMAGES_DIR, filename)
                file.save(img_path)
            else:
                error = "Định dạng ảnh không hợp lệ."
        else:
            error = "Vui lòng tải ảnh hoặc nhập liên kết."

        if img_path and not error:
            bbox = detect_fruit_with_yolo(img_path)
            yolo_image_filename = draw_yolo_bounding_box(img_path, bbox)
            freshness_class_result, freshness_prob_result = predict_fruit_freshness(img_path)
            dominant_colors_info, kmean_image_filename = color_of_image(img_path, bounding_box=bbox)
            color_ripeness_result = name_main_color(dominant_colors_info)
            pytorch_prediction, pytorch_confidence = predict_ripeness_pytorch(img_path)

            predictions = {
                "freshness_class1": freshness_class_result[0], "freshness_prob1": freshness_prob_result[0],
                "freshness_class2": freshness_class_result[1], "freshness_prob2": freshness_prob_result[1],
                "freshness_class3": freshness_class_result[2], "freshness_prob3": freshness_prob_result[2],
                "color_ripeness": color_ripeness_result,
                "pytorch_prediction": pytorch_prediction, "pytorch_confidence": pytorch_confidence
            }

            return render_template("success.html",
                                   img=kmean_image_filename,
                                   yolo_img=yolo_image_filename,
                                   predictions=predictions)
    except Exception as e:
        print(f"Lỗi trong route /success: {e}")
        error = "Đã xảy ra lỗi trong quá trình xử lý."

    return render_template("index.html", error=error)


if __name__ == '__main__':
    app.run(debug=True)
