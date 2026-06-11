#!/bin/bash
#SBATCH --partition=fit
#SBATCH --nodelist=m3u000,m3u001,m3u002,m3u003,m3u004,m3u005,m3u006,m3u007,m3u008
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --account=mg61
#SBATCH --job-name=CIRR
#SBATCH --output=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/logs/%x-%j.out
#SBATCH --error=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/logs/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

source ~/.bashrc
mamba activate tmrl

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6

cd /home/thuy0050/code/vlmcl/src/tevatron

ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl
MODEL_NAME_OR_PATH=openai/clip-vit-large-patch14
EXP_NAME=CCLIP_MSCOCO_i2t
OUTPUT_DIR=$ROOT_DIR/$MODEL_NAME_OR_PATH/$EXP_NAME
mkdir -p "$OUTPUT_DIR"

WANDB_PROJECT=${WANDB_PROJECT:-vlmcl}
export WANDB_PROJECT
export WANDB_NAME="${EXP_NAME}"

# One script for both modes:
#   TRAIN_MODE=single sbatch train.sh
#   TRAIN_MODE=multi  sbatch train.sh
TRAIN_MODE=${TRAIN_MODE:-single}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}

LAUNCHER="python"
if [[ "${TRAIN_MODE}" == "multi" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCHER="torchrun --standalone --nproc_per_node=${NPROC_PER_NODE}"
fi

TRAIN_SUBSETS=(
  "CIRR" # I + T -> I
  "MSCOCO_i2t" # I -> T
  "MSCOCO_t2i" # T -> I
  # "VisDial" # T -> I
  # "WebQA" # T -> I + T
  # "NIGHTS" # I -> I Consider remove due to single modality
  # "VisualNews_i2t" # I -> T Consider remove due to high zero-shot performance
  # "VisualNews_t2i" # T -> I Consider remove due to high zero-shot performance
  # "FashionIQ" # OOD
  # "OVEN" # OOD
  # "Wiki-SS-NQ" # OOD
  # "EDIS" # OOD, T -> I +T Consider remove due to high zero-shot performance
)

ulimit -n 8192 && ${LAUNCHER} hyperbolic/train.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --lora \
  --lora_merge_coeff 0.5 \
  --old_checkpoint_path "$ROOT_DIR/$MODEL_NAME_OR_PATH/CCLIP_CIRR" \
  --subset_name "$subset" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$EXP_NAME" \
  --num_train_epochs 10 \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 1024 \
  --grad_cache True \
  --gc_q_chunk_size 256 \
  --gc_p_chunk_size 256 \
  --dataloader_num_workers 8 \
  --old_embedding_cache_path "$OUTPUT_DIR/old_embedding_cache.pt" \
  --torch_compile True \
  --report_to wandb

  # --old_checkpoint_path "openai/clip-vit-large-patch14" \