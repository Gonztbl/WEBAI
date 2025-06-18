import random, math, uuid
from pathlib import Path
from tkinter import filedialog, simpledialog, messagebox, Tk

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm


# ------------------- Các phép biến đổi ------------------- #
def rotate(img):
    angle = random.uniform(-30, 30)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def flip(img):
    return ImageOps.mirror(img) if random.random() < 0.5 else ImageOps.flip(img)

def to_gray(img):
    return ImageOps.grayscale(img).convert("RGB")

def color_jitter(img):
    brightness = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    contrast   = ImageEnhance.Contrast(brightness).enhance(random.uniform(0.8, 1.2))
    return contrast

def add_noise(img, var=0.01):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, math.sqrt(var), arr.shape)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))

AUG_FUNCS = [rotate, flip, to_gray, color_jitter, add_noise]


# ------------------- Tăng cường ảnh ------------------- #
def augment_images(folder_path, target_total, size=640):
    input_dir = Path(folder_path)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    imgs = []
    for ext in exts:
        imgs.extend(input_dir.glob(ext))
    n_original = len(imgs)

    if n_original == 0:
        messagebox.showerror("Lỗi", "Không có ảnh nào trong thư mục.")
        return

    n_needed = max(0, target_total - n_original)
    if n_needed == 0:
        messagebox.showinfo("Thông báo", "Số lượng ảnh đã đủ, không cần tăng cường thêm.")
        return

    counter = 0
    with tqdm(total=n_needed, desc="Tăng cường ảnh") as pbar:
        while counter < n_needed:
            src_path = random.choice(imgs)
            stem = src_path.stem
            img = Image.open(src_path).convert("RGB")

            n_transforms = random.randint(1, 3)
            transforms = random.sample(AUG_FUNCS, k=n_transforms)
            aug_img = img
            for f in transforms:
                aug_img = f(aug_img)

            aug_img = aug_img.resize((size, size))
            out_name = f"{stem}_aug_{uuid.uuid4().hex[:6]}.jpg"
            out_path = input_dir / out_name
            aug_img.save(out_path, quality=95, optimize=True)

            counter += 1
            pbar.update(1)

    messagebox.showinfo("Hoàn tất", f"Đã tạo {n_needed} ảnh tăng cường mới.")


# ------------------- Giao diện người dùng ------------------- #
def main():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính

    folder_path = filedialog.askdirectory(title="Chọn thư mục ảnh để tăng cường")
    if not folder_path:
        return

    target = simpledialog.askinteger("Số lượng ảnh", "Bạn muốn có tổng cộng bao nhiêu ảnh (kể cả ảnh gốc)?", minvalue=1)
    if not target:
        return

    augment_images(folder_path, target, size=640)


if __name__ == "__main__":
    main()
