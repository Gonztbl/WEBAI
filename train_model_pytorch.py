import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================================
# CẤU HÌNH
# ================================
CONFIG = {
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 25,
    "LEARNING_RATE": 1e-4,
    "DATASET_DIR": "dataset/train",
    "MODEL_SAVE_PATH": "model/fruit_state_classifier.pt",
    "CLASS_INDICES_PATH": "model/fruit_class_indices.json",
    "PATIENCE": 5
}

os.makedirs(os.path.dirname(CONFIG["MODEL_SAVE_PATH"]), exist_ok=True)

# ================================
# TẢI DỮ LIỆU VỚI AUGMENTATION
# ================================
transform = transforms.Compose([
    transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(CONFIG["DATASET_DIR"], transform=transform)

# Lưu class_indices
class_indices = {v: k for k, v in dataset.class_to_idx.items()}
with open(CONFIG["CLASS_INDICES_PATH"], 'w') as f:
    json.dump(class_indices, f, indent=4)
print(f"✅ Đã lưu class indices vào: {CONFIG['CLASS_INDICES_PATH']}")

# Chia tập train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=CONFIG["BATCH_SIZE"])

# ================================
# MÔ HÌNH MOBILENETV2
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

# Thay đầu ra
num_classes = len(dataset.classes)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)
model = model.to(device)

# ================================
# HUẤN LUYỆN
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])

best_val_loss = float('inf')
patience_counter = 0
train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

print("\n🚀 Bắt đầu huấn luyện...")

for epoch in range(CONFIG["EPOCHS"]):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = correct / total
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # Đánh giá
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    print(f"✅ Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
        print(f"📦 Đã lưu mô hình tốt nhất vào: {CONFIG['MODEL_SAVE_PATH']}")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["PATIENCE"]:
            print("🛑 Dừng sớm vì không cải thiện.")
            break

# ================================
# VẼ ĐỒ THỊ
# ================================
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
