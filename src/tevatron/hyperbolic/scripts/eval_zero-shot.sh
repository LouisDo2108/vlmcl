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

MODEL_NAME_OR_PATH=openai/clip-vit-large-patch14
EXP_NAME=zero-shot

ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl
OUTPUT_DIR=$ROOT_DIR/$MODEL_NAME_OR_PATH/$EXP_NAME

mkdir -p "$OUTPUT_DIR"

EVAL_SUBSETS=(
  "CIRR" # I + T -> I
  "MSCOCO_i2t" # I -> T
  "MSCOCO_t2i" # T -> I
  "VisDial" # T -> I
  "WebQA" # T -> I + T
  "NIGHTS" # I -> I Consider remove due to single modality
  "VisualNews_i2t" # I -> T Consider remove due to high zero-shot performance
  "VisualNews_t2i" # T -> I Consider remove due to high zero-shot performance
  "FashionIQ" # OOD
  "OVEN" # OOD
  "Wiki-SS-NQ" # OOD
  "EDIS" # OOD, T -> I +T Consider remove due to high zero-shot performance
)

LAUNCHER="python"
ulimit -n 8192 && ${LAUNCHER} hyperbolic/eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --image_dir /home/thuy0050/mg61_scratch2/thuy0050/data/MMEB/MMEB-eval/image-tasks \
  --subset_name "${EVAL_SUBSETS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$EXP_NAME"