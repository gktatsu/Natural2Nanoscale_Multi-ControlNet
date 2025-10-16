#!/bin/bash
#SBATCH --job-name=n2n_train
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ログディレクトリ作成
mkdir -p logs

echo "Job started on $(date)"
echo "Node: $(hostname)"

# 実行
srun enroot start --rw \
  --mount /home/ul/ul_student/ul_xbf69/tatsuki/Natural2Nanoscale:/workspace \
  --mount /home/ul/ul_student/ul_xbf69/tatsuki/datasets:/datasets \
  hannahkniesel+natural2nanoscale+latest \
  bash -c 'cd /workspace && python train.py \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --image_path "/datasets/Natural2Nanoscale/Hannah/Data/Dataset 1/WannerFIB/WannerFIB_train_images" \
    --mask_path "/datasets/Natural2Nanoscale/Hannah/Data/Dataset 1/WannerFIB/WannerFIB_train_masks" \
    --resume ./models/control_sd15_ini.ckpt \
    --gpus 1 \
    --precision 32 \
    --wandb_api_key 91ca033bf4cedebb62502c72c3f5196cb8940574'

echo "Job finished on $(date)"