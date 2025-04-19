# mask 생성 및 저장
import numpy as np

mask = np.zeros((14, 14))
mask[5:9, :] = 1  # 마스킹할 영역 지정
np.save("mask_tensor/custom_mask.npy", mask)  # → ./custom_mask.npy 로 저장
