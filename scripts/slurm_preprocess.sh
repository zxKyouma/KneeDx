#!/bin/bash
# NOTE: create logs/ before sbatch (SBATCH output paths are relative to submit dir)
#SBATCH --job-name=knee-prep
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-prep-%j.out
#SBATCH --error=logs/slurm-prep-%j.out

set -uo pipefail
ROOT="${RSNA_ROOT:?set RSNA_ROOT to the repo root}"   # EXAMPLE: adapt module/conda lines to your cluster
mkdir -p "$ROOT/logs"
cd "$ROOT"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CONDA_MODULE:-anaconda}"
eval "$(conda shell.bash hook)"
conda activate "${RSNA_CONDA_ENV:-$ROOT/envs/knee}"

PREFIX=${PREFIX:-train_series/}
OUT=${OUT:-data/volumes/train}
META=${META:-data/processed/series_meta_train.parquet}
SIZE=${SIZE:-384}

python src/preprocess.py \
    --zip data/raw/archive.zip \
    --prefix "$PREFIX" \
    --out "$OUT" \
    --meta-out "$META" \
    --size "$SIZE" \
    --workers 32

echo "volumes on disk:"
du -sh "$OUT"
