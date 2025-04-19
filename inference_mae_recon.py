import os
import argparse

import torch
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image, ImageDraw

import numpy as np

from tqdm import tqdm

from models_mae import mae_vit_base_patch16
from util.pos_embed import interpolate_pos_embed

# ----- argparse로 경로 설정 -----
parser = argparse.ArgumentParser(description='MAE 이미지 복원')
parser.add_argument('--model_ckpt', type=str, required=True, help='모델 체크포인트 경로')
parser.add_argument('--image_dir', type=str, required=True, help='입력 이미지 폴더 경로')
parser.add_argument('--output_dir', type=str, required=True, help='결과 이미지 저장 경로')

parser.add_argument('--mask_type', type=str, required=True, help='마스킹 방법, ["random_masking", "center_block", "random_block", "custom_tensor"]')
parser.add_argument('--mask_ratio', type=float, default = 0.75, required=False, help='"random_masking" 선택 시 마스킹 비율')
parser.add_argument('--block_size', type=int, default = 4, required=False, help='"center_block" 선택 시 한 변의 길이')
parser.add_argument('--mask_tensor', type=str, required=False, help='"custom_tensor" 선택 시 mask_tensor 저장된 .npy 경로')

args = parser.parse_args()

if args.mask_type == "random_masking" and args.mask_ratio is None:
    parser.error("--mask_ratio is required for random_masking")
elif args.mask_type == "center_block" and args.block_size is None:
    parser.error("--block_size is required for center_block")
elif args.mask_type == "custom_tensor" and args.mask_tensor is None:
    parser.error("--mask_tensor is required for custom_tensor")

model_ckpt = args.model_ckpt
image_dir = args.image_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# ----- 모델 불러오기 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mae_vit_base_patch16()
checkpoint = torch.load(model_ckpt, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()

# ----- 이미지 전처리 작업 설정 -----
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- 이미지 전처리 -----
def preprocess_image(img_path):
    img_pil = Image.open(img_path).convert('RGB')
    img_pil_resized = img_pil.resize((224, 224), Image.BICUBIC)  # 정규화 전 저장용
    img_tensor = transform(img_pil)  # ← transform 적용
    return img_tensor.unsqueeze(0).to(device), img_pil_resized

# ---- 정규화 복원 -----
def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean

def apply_patch_mask_pil(orig_img_pil, mask, patch_size=16):
    """
    orig_img_pil: PIL.Image (224x224), RGB
    mask: torch.Tensor, shape [1, L], 0: keep, 1: mask
    """
    draw = ImageDraw.Draw(orig_img_pil)
    mask = mask.detach().cpu().numpy()[0]  # shape: (L,)
    
    h = w = 224 // patch_size
    for i, m in enumerate(mask):
        if m == 1:
            row = i // w
            col = i % w
            x0 = col * patch_size
            y0 = row * patch_size
            x1 = x0 + patch_size
            y1 = y0 + patch_size
            draw.rectangle([x0, y0, x1, y1], fill=(127, 127, 127))  # 회색 블럭

    return orig_img_pil

# ----- 복원 수행 -----
def reconstruct_image(img_path):
    global loss_log

    img_tensor, img_pil = preprocess_image(img_path)  # PIL 이미지 같이 받기

    with torch.no_grad():
        kwargs = {}

        if args.mask_type == "center_block":
            kwargs['block_size'] = args.block_size
        elif args.mask_type == "custom_tensor":
            mask_tensor = torch.tensor(np.load(args.mask_tensor)).float()  # [14,14] 또는 [N,14,14]
            kwargs["mask_tensor"] = mask_tensor

        loss, y, mask = model(img_tensor, mask_type = args.mask_type, mask_ratio = args.mask_ratio, **kwargs)
    
    loss_log.append(f"{img_path} 에 대한 loss: {loss.item():.8f}")

    y = model.unpatchify(y)
    y = y.permute(0, 2, 3, 1).detach().cpu().squeeze()

    mask = mask.detach()  # shape: [1, 196]
    mask_for_viz = mask.clone()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)
    mask = mask.permute(0, 2, 3, 1).detach().cpu().squeeze()
    
    # masked image 생성 (PIL 기반)
    masked_img_pil = apply_patch_mask_pil(img_pil.copy(), mask_for_viz)

    base = os.path.splitext(os.path.basename(img_path))[0]
    y = unnormalize(y.permute(2, 0, 1))  # [C, H, W]

    # ----- 마스킹된 부분만 복원값으로 채운 이미지 생성 -----
    y_c, y_h, y_w = y.shape
    orig = unnormalize(img_tensor.squeeze().cpu())  # [3, 224, 224]
    visible = orig.clone()

    # mask: [224, 224, 3] → 다시 [3, 224, 224]로 변환
    mask = mask.permute(2, 0, 1)  # [3, H, W], 0 또는 1

    # 마스킹된 영역(1)인 부분만 복원값 사용
    visible = visible * (1 - mask) + y * mask
    
    save_image(y, os.path.join(output_dir, f"{base}_recon.png"))
    masked_img_pil.save(os.path.join(output_dir, f"{base}_masked.png"))  # 🔄 수정됨
    img_pil.save(os.path.join(output_dir, f"{base}_orig.png"))
    save_image(visible, os.path.join(output_dir, f"{base}_visible.png"))

# 폴더 내 모든 이미지 reconstruct
loss_log = []

for fname in tqdm(os.listdir(image_dir), desc="Reconstructing"):
    if fname.endswith('.png'):
        reconstruct_image(os.path.join(image_dir, fname))

# 각 이미지에 대한 loss를 txt 파일로 저장
with open(os.path.join(output_dir, "images_loss.txt"), "w") as f:
    f.write("\n".join(loss_log))

# ----- 평균 loss 계산 및 출력 -----
# 문자열에서 loss 값만 추출해서 float 리스트로 변환
loss_values = [float(line.split("loss:")[-1].strip()) for line in loss_log]
mean_loss = sum(loss_values) / len(loss_values)
print(f"mean loss: {mean_loss:.8f}")

# 평균 loss도 같이 파일에 저장
with open(os.path.join(output_dir, "images_loss.txt"), "a") as f:
    f.write(f"\nmean loss: {mean_loss:.8f}\n")