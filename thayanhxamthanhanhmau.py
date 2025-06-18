import os
import random
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox, Tk

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm


def is_grayscale(img, threshold=5):
    """Kiểm tra xem ảnh có phải ảnh xám không (R ≈ G ≈ B)."""
    arr = np.array(img.convert("RGB"))
    if len(arr.shape) != 3 or arr.shape[2] != 3:
        return True
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
    diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2])
    return (np.mean(diff_rg) < threshold) and (np.mean(diff_rb) < threshold)


def augment_color_image(img, size=640):
    """Tăng cường ảnh màu bằng các phép biến đổi cơ bản."""
    def color_jitter(img):
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        return img

    def rotate(img):
        return img.rotate(random.uniform(-20, 20), resample=Image.BICUBIC, expand=True)

    def flip(img):
        return ImageOps.mirror(img) if random.random() < 0.5 else ImageOps.flip(img)

    transforms = [color_jitter, rotate, flip]
    random.shuffle(transforms)

    aug_img = img.copy()
    for t in transforms[:random.randint(1, 3)]:
        aug_img = t(aug_img)

    return aug_img.resize((size, size))


def replace_grayscale_images(folder_path, img_size=640):
    folder = Path(folder_path)
    image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    gray_images = []
    color_images = []

    print("🔍 Đang kiểm tra ảnh xám...")
    for path in tqdm(image_paths):
        img = Image.open(path).convert("RGB")
        if is_grayscale(img):
            gray_images.append(path)
        else:
            color_images.append(path)

    print(f"🖤 Số ảnh xám phát hiện: {len(gray_images)}")
    print(f"🎨 Số ảnh màu dùng để thay thế: {len(color_images)}")

    if not gray_images:
        messagebox.showinfo("Thông báo", "Không tìm thấy ảnh xám nào trong thư mục.")
        return

    if not color_images:
        messagebox.showerror("Lỗi", "Không có ảnh màu để thay thế.")
        return

    for gray_path in tqdm(gray_images, desc="Thay thế ảnh xám"):
        # Xoá ảnh xám
        gray_path.unlink()

        # Chọn ảnh màu và tạo ảnh mới
        source_img_path = random.choice(color_images)
        source_img = Image.open(source_img_path).convert("RGB")
        new_img = augment_color_image(source_img, size=img_size)

        new_name = f"{source_img_path.stem}_replaced_{uuid.uuid4().hex[:6]}.jpg"
        new_img.save(folder / new_name, quality=95, optimize=True)

    messagebox.showinfo("Hoàn tất", f"Đã thay thế {len(gray_images)} ảnh xám bằng ảnh màu mới.")


def main():
    root = Tk()
    root.withdraw()

    folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
    if not folder:
        return

    replace_grayscale_images(folder)


if __name__ == "__main__":
    main()
