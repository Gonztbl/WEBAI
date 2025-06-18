import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ====================================================================
# --- 1. CẤU HÌNH TRUNG TÂM ---
# ====================================================================
CONFIG = {
    "IMG_WIDTH": 224,
    "IMG_HEIGHT": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 25,  # Có thể tăng lên vì đã có EarlyStopping
    "LEARNING_RATE": 0.0001,
    "DATASET_DIR": 'dataset',
    "MODEL_SAVE_PATH": 'model/fruit_state_classifier.keras',  # Lưu ở định dạng .keras
    "CLASS_INDICES_PATH": 'model/fruit_class_indices.json'
}

# Tạo thư mục lưu model nếu chưa tồn tại
os.makedirs(os.path.dirname(CONFIG["MODEL_SAVE_PATH"]), exist_ok=True)


# ====================================================================
# --- 2. HÀM CHUẨN BỊ DỮ LIỆU ---
# ====================================================================
def create_data_generators():
    """Tạo ra các data generator cho tập train và validation."""
    train_dir = os.path.join(CONFIG["DATASET_DIR"], 'train')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Lưu lại class_indices
    class_indices = train_generator.class_indices
    class_indices_rev = {v: k for k, v in class_indices.items()}
    with open(CONFIG["CLASS_INDICES_PATH"], 'w') as f:
        json.dump(class_indices_rev, f, indent=4)
    print(f"Đã lưu class indices vào: {CONFIG['CLASS_INDICES_PATH']}")

    return train_generator, validation_generator

    # Lưu lại class_indices
    class_indices = train_generator.class_indices
    # Đảo ngược key và value để dễ tra cứu từ index -> tên lớp
    class_indices_rev = {v: k for k, v in class_indices.items()}
    with open(CONFIG["CLASS_INDICES_PATH"], 'w') as f:
        json.dump(class_indices_rev, f, indent=4)
    print(f"Đã lưu class indices vào: {CONFIG['CLASS_INDICES_PATH']}")

    return train_generator, validation_generator


# ====================================================================
# --- 3. HÀM XÂY DỰNG MODEL ---
# ====================================================================
def build_model(num_classes):
    """Xây dựng model MobileNetV2 cho transfer learning."""
    base_model = MobileNetV2(
        input_shape=(CONFIG["IMG_WIDTH"], CONFIG["IMG_HEIGHT"], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Đóng băng base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Thêm Dropout để giảm overfitting
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ====================================================================
# --- 4. HÀM VẼ ĐỒ THỊ KẾT QUẢ ---
# ====================================================================
def plot_history(history):
    """Vẽ đồ thị accuracy và loss từ history object."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('training_history.png')
    plt.show()


# ====================================================================
# --- 5. HÀM CHÍNH ĐỂ HUẤN LUYỆN ---
# ====================================================================
if __name__ == '__main__':
    # Bước 1: Chuẩn bị dữ liệu
    train_generator, validation_generator = create_data_generators()

    # Bước 2: Xây dựng model
    model = build_model(num_classes=train_generator.num_classes)
    model.summary()

    # Bước 3: Định nghĩa các Callbacks
    # Lưu model tốt nhất dựa trên val_accuracy
    checkpoint = ModelCheckpoint(
        filepath=CONFIG["MODEL_SAVE_PATH"],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    # Dừng sớm nếu không có cải thiện
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Dừng nếu val_loss không cải thiện sau 5 epochs
        restore_best_weights=True,
        verbose=1
    )
    # Giảm learning rate khi chững lại
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1
    )

    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # Bước 4: Huấn luyện model
    print("\nBắt đầu quá trình huấn luyện...")
    history = model.fit(
        train_generator,
        epochs=CONFIG["EPOCHS"],
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // CONFIG["BATCH_SIZE"],
        validation_steps=validation_generator.samples // CONFIG["BATCH_SIZE"],
        callbacks=callbacks_list
    )

    # Bước 5: Vẽ đồ thị
    print("\nHoàn tất huấn luyện. Đang vẽ đồ thị kết quả...")
    plot_history(history)
