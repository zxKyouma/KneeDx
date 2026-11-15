#!/bin/bash
# Export one fold's out-of-fold predictions, immediately after that fold trains.
#
# The 5-fold launched from a train.py predating in-loop OOF saving, so every
# fold needs this second inference pass - `grep -c "OOF preds"` on their logs
# returns 0. Self-training and the extraction referee both consume the result,
# so this is on the critical path, not a nice-to-have.
#
# Deliberately SHORT walltime. This is ~870 studies of inference, ~20 minutes,
# against 16-hour training jobs sitting in the same queue. A short limit lets
# SLURM's backfill scheduler slot it into gaps instead of making the highest-
# value work wait behind ten exploratory arms. Submit with
# --dependency=afterok:<fold job> so it fires the moment the fold finishes.
#
#   FOLD=0 EXPECT=919 sbatch --dependency=afterok:58814847 scripts/slurm_export_oof.sh
#
# NOTE: create logs/ before sbatch (SBATCH output paths are relative to submit dir)
#SBATCH --job-name=knee-oof
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-oof-%j.out
#SBATCH --error=logs/slurm-oof-%j.out

set -uo pipefail
ROOT="${RSNA_ROOT:?set RSNA_ROOT to the repo root}"   # EXAMPLE: adapt module/conda lines to your cluster
mkdir -p "$ROOT/logs"
cd "$ROOT"
export HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CONDA_MODULE:-anaconda}"
eval "$(conda shell.bash hook)"
conda activate "${RSNA_CONDA_ENV:-$ROOT/envs/knee}"
nvidia-smi --query-gpu=name --format=csv,noheader

FOLD=${FOLD:?set FOLD}
DIR=${DIR:-checkpoints/f5_convnext_tiny}
# EMA weights, because they score better on gold in every arm measured and
# better predictions are better pseudo-labels. The ema file stores the averaged
# weights under "model" (train.py loads them in before saving), so nothing
# special is needed to read them.
CKPT=${CKPT:-$DIR/fold${FOLD}_ema.pt}

# --expect-val makes a split that fails to reproduce abort instead of silently
# emitting in-sample predictions dressed up as out-of-fold ones.
python src/export_oof.py \
    --ckpt "$CKPT" \
    --expect-val "${EXPECT:?set EXPECT from the val count in the training log}" \
    --workers 8

echo "oof files now present:"
ls -la "$DIR"/*oof*.npz 2>/dev/null || echo "  none"
