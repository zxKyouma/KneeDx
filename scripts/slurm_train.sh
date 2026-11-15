#!/bin/bash
# NOTE: create logs/ before sbatch (SBATCH output paths are relative to submit dir)
#SBATCH --job-name=knee-train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-train-%j.out
#SBATCH --error=logs/slurm-train-%j.out

set -uo pipefail
ROOT="${RSNA_ROOT:?set RSNA_ROOT to the repo root}"   # EXAMPLE: adapt module/conda lines to your cluster
mkdir -p "$ROOT/logs"
cd "$ROOT"
export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1          # backbone weights are pre-cached on the login node
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CONDA_MODULE:-anaconda}"
eval "$(conda shell.bash hook)"
conda activate "${RSNA_CONDA_ENV:-$ROOT/envs/knee}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Full 384 frame, no crop: measured tissue bbox is 327x364 inside the 384
# frame, so a 320 crop was cutting anatomy off 86% of series. 24 slices instead
# of 16 lifts median coverage from 53% to ~80% of the acquired slices, which
# matters for focal findings (a meniscal tear spans 1-2 slices).
# 205 MB/image at 384, so bs=1 x 4 series x 24 slices = 96 images ~= 20 GB.
# Effective batch stays 8 via accumulation.
python src/train.py \
    --fold "${FOLD:-0}" \
    --backbone "${BACKBONE:-convnext_tiny}" \
    --epochs "${EPOCHS:-12}" \
    --bs 1 --accum 8 \
    --n-series 6 --n-slices 24 --size 384 \
    --workers 8 \
    --out "checkpoints/${BACKBONE:-convnext_tiny}"
