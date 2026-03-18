# MAE_custom
This repository contains code for training and image reconstruction with an MAE model using `mae_vit_large_patch16` as the backbone.  
It provides four custom masking options: `random_masking`, `center_block`, `random_block`, and `custom_tensor`.


## Environment setting
Please set up the environment by following the instructions in the official MAE repository: https://github.com/facebookresearch/mae.git
Make sure to use `timm==0.3.2`.


## Data Preparation
Organize the image data in the following structure:
```
| data
    | -- train / 0
        | -- 00001_image_000001.png
        | -- 00001_image_000002.png
        ...
    | -- val / 0
        | -- 00004_image_000001.png
        | -- 00004_image_000002.png
        ...
```
That is, the images should be placed under `train/class_0` and `val/class_0`.


## Training
The script for training the image reconstruction task is `main_pretrain.py`.

First, download the official pretrained weight using the following command.
This weight was trained on ImageNet with the random masking strategy.
```
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
```

Then run training as follows.
The hyperparameters are specified in the command below, and training logs as well as checkpoint files (.pth) for each epoch will be saved during training.
```
python main_pretrain.py \
  --model mae_vit_large_patch16 \
  --input_size 224 \
  --batch_size 32 \
  --mask_ratio 0.75 \
  --epochs 200 \
  --warmup_epochs 0 \
  --blr 1e-4 \
  --weight_decay 0.05 \
  --data_path [path to training data] \
  --resume [path to the downloaded mae_visualize_vit_large.pth] \
  --mask_type [masking type]
```

If the training images are stored under `data/original/train/class_0/image.png`,
then `data/original/` should be passed as `--data_path`.

The available options for `--mask_type` are as follows:
  
`random_masking`: Generates a random mask. This option should be used together with mask_ratio.  

`center_block`: Generates a mask at the center of the image. The default side length is 4, and it can be changed with the block_size option.

`random_block`: Generates a random rectangular mask.

`custom_tensor`: Loads a predefined 14×14 mask tensor from a .npy file. In this case, the path to the .npy file must also be provided through mask_tensor.


## Inference
Use `inference_mae_recon.py` to perform image reconstruction with a trained model.

For each input image, the reconstruction loss is saved as a .txt file.
In addition, the following outputs are generated: input image, masked image, reconstructed image, and visible image(where the visible patches in the masked image are combined with the reconstructed image)

```
cd custom_mae
python inference_mae.py \
  --model_ckpt [path to checkpoint] \
  --image_dir [path to input images] \
  --output_dir [path to save results] \
  --mask_type [masking type]
```


## Evaluation
You can use `metric.py` to measure PSNR, SSIM, and MSE for the generated images.
Use `_orig.png` as the ground-truth image to evaluate the metrics of `_masked.png` and `_visible.png`.


## Acknowledgement
This repository is based on the official MAE repository: https://github.com/facebookresearch/mae.git
