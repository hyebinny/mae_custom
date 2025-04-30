# mask 생성 및 저장
import numpy as np

# ROI: 위에서부터 세로 1:13, 왼쪽에서부터 가로 3:11

# ROI
mask = np.zeros((14, 14))
mask[1:13, 4:10] = 1  
np.save("mask_tensor/roi.npy", mask)  
print(mask)

# 좌측 전신
mask = np.zeros((14, 14))
mask[1:13, 3:6] = 1  
np.save("mask_tensor/left_all.npy", mask)  
print(mask)

# 우측 전신
mask = np.zeros((14, 14))
mask[1:13, 6:10] = 1  
np.save("mask_tensor/right_all.npy", mask)  
print(mask)

# 하체 전신
mask = np.zeros((14, 14))
mask[7:13, 4:10] = 1  
np.save("mask_tensor/bottom_all.npy", mask) 
print(mask)

# 상체 전신
mask = np.zeros((14, 14))
mask[1:7, 4:10] = 1 
np.save("mask_tensor/top_all.npy", mask)  
print(mask)

