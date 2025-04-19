import os
import shutil

# 경로 설정(original, train)
src_root = "/mnt/d/DL-proj/data/preprocessed_data/rgb_data_100"
split_txt = "/mnt/d/DL-proj/mae_custom/trainval_split/train.txt"
dst_root = "/mnt/d/DL-proj/data/original/train"

# 경로 설정(original, val)
# src_root = "/mnt/d/DL-proj/data/preprocessed_data/rgb_data_100"
# split_txt = "/mnt/d/DL-proj/mae_custom/trainval_split/val.txt"
# dst_root = "/mnt/d/DL-proj/data/original/train"


# 저장 경로 생성
os.makedirs(dst_root, exist_ok=True)

# val.txt에서 폴더 번호 읽기
with open(split_txt, "r") as f:
    folder_list = [line.strip() for line in f.readlines()]

# 각 폴더의 이미지 복사
for folder in folder_list:
    folder_path = os.path.join(src_root, folder)
    if not os.path.exists(folder_path):
        continue  # 폴더가 없으면 건너뜀

    for img_name in os.listdir(folder_path):
        src_img = os.path.join(folder_path, img_name)
        dst_img_name = f"{folder}_{img_name}"
        dst_img = os.path.join(dst_root, dst_img_name)
        shutil.copy(src_img, dst_img)
