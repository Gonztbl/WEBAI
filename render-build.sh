#!/usr/bin/env bash
# Thoát ngay khi có lỗi
set -o errexit

echo "--- Bắt đầu kịch bản build tùy chỉnh ---"

# Cài đặt Git LFS
echo "--- Cài đặt Git LFS ---"
apt-get update
apt-get install -y git-lfs

# Lệnh quan trọng nhất: Kéo các tệp lớn từ Git LFS
echo "--- Tải về các tệp mô hình lớn (git lfs pull) ---"
git lfs pull

# Cài đặt các thư viện Python
echo "--- Cài đặt các gói Python từ requirements.txt ---"
pip install -r requirements.txt

echo "--- Kịch bản build hoàn tất ---"