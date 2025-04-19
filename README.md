# MAE_custom
MAE에서 image reconstruction까지만의 과정을 다룹니다.
mae_vit_base_patch16만의 적용을 다룹니다. 
mae_vit_base_patch16는 한 패치의 크기가 (16픽셀X16픽셀)인 경우로, 224X224 크기의 이미지가 들어왔을 때 한 변에 14개의 패치로 분할하게 됩니다.

본 repo에서는 4가지의 custom 옵션만을 제공합니다: `random_masking`, `center_block`, `random_block`, 그리고 `custom_tensor`.


## Environment setting
https://github.com/facebookresearch/mae.git 에 안내되어 있는 내용을 따라 환경을 설정해주세요.
이때 `timm==0.3.2` 버전은 꼭 맞춰주셔아 합니다! (버전이 다르면 모델이 안 돌아갑니다.)

제가 작업한 환경은 아래와 같습니다.

WSL
ubuntu: 24.04
codna: 24.9.2
CUDA: 11.6
TORCH: 1.13.1+cu116 (cuda와 torch의 cuda 버전을 꼭 맞춰주셔야 합니다.)

base 환경의 CUDA 버전이 다르면, 가상환경 생성 후 아래 명령어로 가상환경 안에 CUDA를 따로 구성하실 수 있습니다.
```
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
```


## Data Preparation
아래와 같은 구조로 이미지 데이터를 조직하세요.
```
| data
		| -- dark
		| -- mix
		| -- original
				| -- train / class_0
						| -- 00001_image_000001.png
						| -- 00001_image_000002.png
						...
						| -- 00090_image_000015.png
				| -- val / class_0
						| -- 00004_image_000001.png
						| -- 00004_image_000002.png
						...
						| -- 00087_image_000015.png
```
즉, 이미지들은 train/class_0 그리고 val/class_0 안에 들어있어야 합니다.
train/val split은 본 repo의 `trainval_split/train_val_copy.py`를 활용하실 수 있습니다.


## Training
Image reconstruction을 위한 script는 `main_pretrain.py`입니다.

먼저 아래 명령어로 official weight을 다운받으세요. (random masking 방식으로 ImageNet에 대해 학습된 weigth)
```wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth```

다음과 같이 학습을 시작할 수 있습니다. 학습 log, 각 epoch별 weight(.pth 파일), 그리고 학습에 사용된 이미지 중 첫 번째 이미지에 대한 시각화 결과가 저장됩니다.
```
python main_pretrain.py \
  --model mae_vit_base_patch16 \
  --input_size 224 \
  --batch_size 32 \
  --mask_ratio 0.75 \
  --epochs [epoch 수 지정] \
  --warmup_epochs 5 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --data_path [학습 데이터 경로] \
  --resume [위에서 다운받은 mae_visualize_vit_base.pth 경로] \
  --mask_type [masking 방법]

```

학습 데이터가 `data/original/train/class_0/image.png` 경로에 저장되어 있다면 학습 데이터 경로는 `data/original/`으로 전달합니다.
`--mask_type`으로 지정할 수 있는 옵션은 다음과 같습니다. `random_masking`, `center_block`, `random_block`, 그리고 `custom_tensor`.
`random_masking`: 랜덤으로 마스킹 생성, `mask_ratrio`와 같이 주어져야 합니다.
`center_block`: 이미지 중앙에 mask를 생성합니다. 한 변의 길이의 기본값은 4이며 `block_size` 옵션으로 변경할 수 있습니다.
`random_block`: 랜덤으로 사각형 모양 마스킹 생성
`custom_tensor`: masking이 기록되어 있는 .npy 파일이 저장된 위치를 불러와 해당 14X14 텐서를 마스킹으로 사용합니다. .npy 파일이 저장된 위치가 `mask_tensor`으로 함께 주어져야 합니다.


## Visualization script
`inference_mae_recon.py`는 inference 결과의 시각화 코드입니다.
각 inference에 사용된 각 이미지별 loss가 .txt 파일로 저장되고,
input image, masked image, reconstructed image, 그리고 masked image 중 visibile patch를 reconstructed image와 조합한 visible image가 생성됩니다.

```
cd custom_mae
python inference_mae_recon.py --model_ckpt [ckpt 경로] \
  --image_dir [input 이미지들이 포함된 경로] \
  --output_dir [결과를 저장할 경로] \
  --mask_type [masking 방법]
```
