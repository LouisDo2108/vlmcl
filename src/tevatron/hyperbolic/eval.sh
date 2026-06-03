#!/bin/bash
#SBATCH --partition=fit
#SBATCH --nodelist=m3u000,m3u001,m3u002,m3u003,m3u004,m3u005,m3u006,m3u007,m3u008
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

MODEL_NAME=openai/clip-vit-large-patch14
EXP_NAME=CIRR-2k-steps-lr3e-5

OUTPUT_BASE_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl
OUTPUT_DIR=$OUTPUT_BASE_DIR/$MODEL_NAME/$EXP_NAME

mkdir -p "$OUTPUT_DIR"



LAUNCHER="python"
# CIRR MSCOCO_i2t MSCOCO_t2i NIGHTS VisDial VisualNews_i2t VisualNews_t2i WebQA
ulimit -n 8192 && ${LAUNCHER} hyperbolic/eval.py \
  --model_name_or_path "$MODEL_NAME" \
  --bf16 \
  --lora \
  --lora_name_or_path "$OUTPUT_DIR" \
  --image_dir /home/thuy0050/ft49_scratch2/thuy0050/data/MMEB/MMEB-eval/image-tasks \
  --subset_name CIRR \
  --per_device_eval_batch_size 256 \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$EXP_NAME"