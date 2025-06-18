# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from PIL import Image
from tqdm import tqdm  # thêm vào đầu file nếu chưa có


# ==============================================================================
# HÀM DỰ ĐOÁN (ĐỊNH NGHĨA NGOÀI CÙNG ĐỂ CÓ THỂ TÁI SỬ DỤNG)
# ==============================================================================
def predict_new_image_pytorch(model, device, image_path, class_names):
    """Hàm tải ảnh, xử lý và dự đoán bằng mô hình PyTorch đã huấn luyện."""
    if not os.path.exists(image_path):
        print(f"LỖI: Không tìm thấy file ảnh tại '{image_path}'")
        return

    # Định nghĩa lại các phép biến đổi cho ảnh đầu vào
    img_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])

    # Đặt mô hình ở chế độ đánh giá
    model.eval()

    # Tải và biến đổi ảnh
    image = Image.open(image_path).convert('RGB')
    image_transformed = inference_transform(image)
    image_batch = image_transformed.unsqueeze(0)
    image_batch = image_batch.to(device)

    with torch.no_grad():
        output = model(image_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = class_names[predicted_idx.item()]

        print(f"\n--- Dự đoán cho ảnh: {os.path.basename(image_path)} ---")
        print(f"Đây có thể là quả '{predicted_class}' với độ tự tin {confidence.item() * 100:.2f}%.")

        # Hiển thị ảnh
        plt.imshow(image)
        plt.title(f"Dự đoán: {predicted_class} ({confidence.item() * 100:.2f}%)")
        plt.axis("off")
        plt.show()


# ==============================================================================
# ĐOẠN CODE CHÍNH - ĐƯỢC BẢO VỆ BỞI if __name__ == '__main__':
# ==============================================================================
if __name__ == '__main__':
    # PHẦN 1: THIẾT LẬP CÁC THAM SỐ VÀ TẢI DỮ LIỆU
    # --------------------------------------------------------------------------

    # THAY ĐỔI ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI MÁY TÍNH CỦA BẠN
    # Dựa trên log lỗi của bạn, tôi giữ nguyên đường dẫn này
    base_dir = 'Fruit_Classification_Data'  # Giả sử bạn chạy script từ thư mục gốc của dự án
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # Kiểm tra xem đường dẫn có tồn tại không
    if not os.path.exists(train_dir):
        print(f"LỖI: Không tìm thấy thư mục huấn luyện tại '{train_dir}'")
        exit()

    # Các tham số
    batch_size = 32
    img_size = 224
    epochs = 25

    # Xác định thiết bị (sử dụng GPU nếu có)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Sẽ sử dụng thiết bị: {device}")

    # Định nghĩa các phép biến đổi và tăng cường dữ liệu
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ]),
    }

    # Tải dữ liệu bằng ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])
    }

    # Tạo các DataLoader
    # QUAN TRỌNG: num_workers>0 là nguyên nhân gây lỗi trên Windows nếu không có __main__
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Đã tìm thấy {num_classes} lớp: {class_names}")

    # PHẦN 2: XÂY DỰNG MÔ HÌNH (SỬ DỤNG TRANSFER LEARNING)
    # --------------------------------------------------------------------------

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    print("\nLớp phân loại mới của mô hình:")
    print(model.classifier)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # PHẦN 3: VIẾT VÒNG LẶP HUẤN LUYỆN
    # --------------------------------------------------------------------------

    print("\nBắt đầu quá trình huấn luyện...")
    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch + 1}/{epochs}", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print()

    time_elapsed = time.time() - start_time
    print(f'Huấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # PHẦN 4: ĐÁNH GIÁ VÀ LƯU MÔ HÌNH
    # --------------------------------------------------------------------------

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history['train_acc'], label='Độ chính xác Huấn luyện')
    plt.plot(range(epochs), history['val_acc'], label='Độ chính xác Kiểm định')
    plt.legend(loc='lower right')
    plt.title('Biểu đồ Độ chính xác')
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history['train_loss'], label='Mất mát Huấn luyện')
    plt.plot(range(epochs), history['val_loss'], label='Mất mát Kiểm định')
    plt.legend(loc='upper right')
    plt.title('Biểu đồ Mất mát')
    plt.show()

    model_filename = 'fruit_ripeness_model_pytorch.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"\nMô hình đã được lưu thành công với tên '{model_filename}'")

    # PHẦN 5: SỬ DỤNG MÔ HÌNH ĐỂ DỰ ĐOÁN ẢNH MỚI
    # --------------------------------------------------------------------------
    # Bỏ comment (xóa dấu #) ở dòng dưới và thay tên file để thử nghiệm
    # predict_new_image_pytorch(model, device, 'ten_file_anh_cua_ban.jpg', class_names)