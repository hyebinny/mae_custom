import torch
from torchvision import transforms
from PIL import Image
import os
from models_mae import mae_vit_base_patch16
from util.pos_embed import interpolate_pos_embed
from torchvision.utils import save_image

# ----- 설정 -----
# model_ckpt = '/mnt/d/DL-proj/mae_custom/ckpt/mae_visualize_vit_base.pth'
# model_ckpt = '/mnt/d/DL-proj/mae_custom/output_recon/checkpoint-30.pth'
model_ckpt = '/mnt/d/DL-proj/mae_custom/output_recon/checkpoint-49.pth'

image_dir = '/mnt/d/DL-proj/data/original/val/class_0'  # 추론할 이미지 폴더
output_dir = '/mnt/d/DL-proj/data/custom_inference_outputs'
os.makedirs(output_dir, exist_ok=True)

# ----- 모델 불러오기 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mae_vit_base_patch16()
checkpoint = torch.load(model_ckpt, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()

# ----- 이미지 전처리 -----
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

from PIL import ImageDraw

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
    img_tensor, img_pil = preprocess_image(img_path)  # PIL 이미지 같이 받기

    with torch.no_grad():
        loss, y, mask = model(img_tensor, mask_ratio=0.75)
    
    print(img_path, "에 대한 loss:", loss.item())


    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu().squeeze()

    mask_for_viz = mask.detach().clone()  # shape: [1, 196]
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu().squeeze()

    img_patch = img_tensor.clone()
    img_patch = torch.einsum('nchw->nhwc', img_patch).detach().cpu().squeeze()
    
    # masked image 생성 (PIL 기반)
    masked_img_pil = apply_patch_mask_pil(img_pil.copy(), mask_for_viz)

    base = os.path.splitext(os.path.basename(img_path))[0]
    y = unnormalize(y.permute(2, 0, 1))  # [C, H, W]

    save_image(y, os.path.join(output_dir, f"{base}_recon.png"))
    masked_img_pil.save(os.path.join(output_dir, f"{base}_masked.png"))  # 🔄 수정됨
    img_pil.save(os.path.join(output_dir, f"{base}_orig.png"))



# 폴더 내 모든 이미지 reconstruct
for fname in os.listdir(image_dir):
    if fname.endswith('.png'):
        reconstruct_image(os.path.join(image_dir, fname))
