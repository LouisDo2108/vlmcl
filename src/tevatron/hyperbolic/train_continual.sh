#!/bin/bash
#SBATCH --partition=fit
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --account=mg61
#SBATCH --job-name=MSCOCO_i2t_10epoch_merge_coeff1.0
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
EXP_NAME=MSCOCO_i2t
OUTPUT_DIR=$ROOT_DIR/$MODEL_NAME_OR_PATH/$EXP_NAME
mkdir -p "$OUTPUT_DIR"

LORA_NAME_OR_PATH=$ROOT_DIR/$MODEL_NAME_OR_PATH/CIRR

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
# for subset in CIRR MSCOCO_i2t MSCOCO_t2i VisDial WebQA; do

for subset in MSCOCO_i2t; do
ulimit -n 8192 && ${LAUNCHER} hyperbolic/train.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --lora \
  --lora_merge_coeff 1.0 \
  --lora_name_or_path "$LORA_NAME_OR_PATH" \
  --subset_name "$subset" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$EXP_NAME" \
  --num_train_epochs 10 \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 1024 \
  --grad_cache True \
  --gc_q_chunk_size 128 \
  --gc_p_chunk_size 128 \
  --dataloader_num_workers 8 \
  --report_to wandb
done

# --lora_name_or_path "$ROOT_DIR/$MODEL_NAME_OR_PATH/CIRR-10epoch_bidirectional_loss" \
# done
# ulimit -n 8192 && ${LAUNCHER} hyperbolic/eval.py \
#   --model_name_or_path "$MODEL_NAME_OR_PATH" \
#   --bf16 \
#   --lora \
#   --lora_merge_coeff 1.0 \
#   --lora_name_or_path "$OUTPUT_DIR" \
#   --image_dir /home/thuy0050/mg61_scratch2/thuy0050/data/MMEB/MMEB-eval/image-tasks \
#   --subset_name CIRR \
#   --per_device_eval_batch_size 256 \
#   --output_dir "$OUTPUT_DIR" \
#   --run_name "$EXP_NAME"