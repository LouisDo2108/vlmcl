#!/bin/bash
#SBATCH --partition=fit
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --account=mg61
#SBATCH --job-name=CIRR-10epoch-lr3e-5_weight_decay1e-2
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

OUTPUT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/clip/CIRR-10epoch-lr3e-5_weight_decay1e-2
mkdir -p "$OUTPUT_DIR"

EXP_NAME=clip-CIRR-10epoch-lr3e-5_weight_decay1e-2
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

# CIRR MSCOCO_i2t MSCOCO_t2i NIGHTS VisDial VisualNews_i2t VisualNews_t2i WebQA
ulimit -n 8192 && ${LAUNCHER} hyperbolic/train.py \
  --model_name_or_path openai/clip-vit-large-patch14 \
  --bf16 \
  --lora \
  --subset_name CIRR \
  --num_sample_per_subset 100000 \
  --image_dir /home/thuy0050/ft49_scratch2/thuy0050/data/MMEB/MMEB-train \
  --max_len 77 \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$EXP_NAME" \
  --logging_steps 10 \
  --num_train_epochs 10 \
  --save_strategy epoch \
  --learning_rate 3e-5 \
  --weight_decay 1e-2 \
  --normalize True \
  --temperature 0.02 \
  --per_device_train_batch_size 1024 \
  --grad_cache True \
  --gc_q_chunk_size 128 \
  --gc_p_chunk_size 128 \
  --dataloader_num_workers 8 \
  --save_safetensors True \
  --remove_unused_columns False \
  --report_to wandb

  # --max_steps 2000 \
  # --warmup_steps 200 \
  # --save_steps 1000 \