import os
import cv2
import glob
import numpy as np
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(infer_dir, output_txt_path):
    visible_files = sorted(glob.glob(os.path.join(infer_dir, '*_visible.png')))
    psnr_list, ssim_list = [], []

    for vis_path in visible_files:
        base_name = os.path.basename(vis_path).replace('_visible.png', '')
        gt_path = os.path.join(infer_dir, f"{base_name}_orig.png")

        if not os.path.exists(gt_path):
            print(f"GT 이미지 없음: {gt_path}")
            continue

        vis_img = cv2.imread(vis_path)
        gt_img = cv2.imread(gt_path)

        if vis_img is None or gt_img is None:
            print(f"이미지 로드 실패: {vis_path} 또는 {gt_path}")
            continue

        if vis_img.shape != gt_img.shape:
            print(f"크기 불일치: {vis_path}")
            continue

        # BGR → RGB 변환 (ssim은 RGB 순서 요구)
        vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        psnr_score = psnr(gt_rgb, vis_rgb, data_range=255)
        ssim_score = ssim(gt_rgb, vis_rgb, data_range=255, channel_axis=2)

        psnr_list.append(psnr_score)
        ssim_list.append(ssim_score)

    with open(output_txt_path, 'w') as f:
        if psnr_list:
            mean_psnr = np.mean(psnr_list)
            mean_ssim = np.mean(ssim_list)
            f.write(f"Mean PSNR: {mean_psnr:.4f}\n")
            f.write(f"Mean SSIM: {mean_ssim:.4f}\n")
            print(f"Mean PSNR: {mean_psnr:.4f}\n")
            print(f"Mean SSIM: {mean_ssim:.4f}\n")
        else:
            f.write("No valid images to evaluate.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_dir', type=str, required=True, help='Path to inference directory')
    parser.add_argument('--output', type=str, default=None, help='Path to output .txt file')
    args = parser.parse_args()

    output_path = args.output if args.output else os.path.join(args.infer_dir, 'metrics_results.txt')
    calculate_metrics(args.infer_dir, output_path)









# def calculate_psnr_all(infer_dir, output_txt_path):
#     visible_files = sorted(glob.glob(os.path.join(infer_dir, '*_visible.png')))
#     psnr_list = []

#     with open(output_txt_path, 'w') as f:
#         for vis_path in visible_files:
#             base_name = os.path.basename(vis_path).replace('_visible.png', '')
#             gt_path = os.path.join(infer_dir, f"{base_name}_orig.png")

#             if not os.path.exists(gt_path):
#                 print(f"GT 이미지 없음: {gt_path}")
#                 continue

#             vis_img = cv2.imread(vis_path)
#             gt_img = cv2.imread(gt_path)

#             if vis_img is None or gt_img is None:
#                 print(f"이미지 로드 실패: {vis_path} 또는 {gt_path}")
#                 continue

#             if vis_img.shape != gt_img.shape:
#                 print(f"크기 불일치: {vis_path}")
#                 continue

#             score = psnr(gt_img, vis_img, data_range=255)
#             psnr_list.append(score)
#             f.write(f"{base_name}.png: {score:.4f}\n")

#         if psnr_list:
#             mean_psnr = np.mean(psnr_list)
#             f.write(f"\nMean PSNR: {mean_psnr:.4f}\n")
#         else:
#             f.write("\nNo valid images to evaluate.\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--infer_dir', type=str, required=True, help='Path to inference directory')
#     parser.add_argument('--output', type=str, default=None, help='Path to output .txt file')

#     args = parser.parse_args()

#     output_path = args.output if args.output else os.path.join(args.infer_dir, 'psnr_results.txt')
#     calculate_psnr_all(args.infer_dir, output_path)
