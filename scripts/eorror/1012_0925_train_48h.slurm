#!/bin/bash
#SBATCH --job-name=n2n_train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ========= モジュール =========
# BwUniClusterの環境に合わせてenrootをロード（名前は環境により異なります）
module purge
module load devel/enroot || module load enroot || true

# ========= 変数定義（必要に応じて調整） =========
IMAGE="hannahkniesel+natural2nanoscale+latest"

# ホスト側パス（ユーザ環境の実ディレクトリ）
HOST_WORKDIR="/home/ul/ul_student/ul_xbf69/tatsuki/Natural2Nanoscale"
HOST_DATA="/home/ul/ul_student/ul_xbf69/tatsuki/datasets"

# コンテナ内でのマウント先
CTR_WORKDIR="/workspace"
CTR_DATA="/datasets"

# WANDB などの環境変数（必要に応じてここに書くと安全）
export WANDB_API_KEY="91ca033bf4cedebb62502c72c3f5196cb8940574"
export PYTHONUNBUFFERED=1

# ログディレクトリ作成
mkdir -p logs

echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "PWD : $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# ========= 実行コマンド（コンテナ内で実行される内容） =========
CTR_CMD=$(cat <<'EOF'
set -euo pipefail
cd /workspace

python train.py \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --image_path "/datasets/Natural2Nanoscale/Original/Dataset 1/Images" \
  --mask_path "/datasets/Natural2Nanoscale/Hannah/Data/Dataset 1/WannerFIB/WannerFIB_train_images" \
  --resume ./models/control_sd15_ini.ckpt \
  --gpus 1 \
  --precision 32 \
  --wandb_api_key "${WANDB_API_KEY}"
EOF
)

# ========= 実行（割り当てられたGPUノード上でenroot起動） =========
# srun で起動して割り当て資源に紐づける（multi-nodeでないので --ntasks=1 のデフォルトでOK）
srun enroot start --rw \
  --mount "${HOST_WORKDIR}:${CTR_WORKDIR}" \
  --mount "${HOST_DATA}:${CTR_DATA}" \
  "${IMAGE}" \
  bash -lc "${CTR_CMD}"

echo "Job finished on $(date)"
