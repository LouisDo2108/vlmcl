#!/bin/bash
#SBATCH --partition=fit
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --account=mg61
#SBATCH --job-name=clip-ft
#SBATCH --output=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/logs/slurm-%x-%j.out
#SBATCH --error=/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/logs/slurm-%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

source ~/.bashrc
mamba activate tmrl

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6

cd /home/thuy0050/code/vlmcl/src/tevatron

OUTPUT_DIR=/home/thuy0050/ft49_scratch2/thuy0050/exp/vlmcl/hyperbolic/clip-vit-large-patch14
mkdir -p "$OUTPUT_DIR"

# Single GPU: use python. Multi-GPU: torchrun --nproc_per_node=N hyperbolic/train.py ...

# ulimit -n 4096 && 
# CIRR MSCOCO_i2t MSCOCO_t2i NIGHTS VisDial VisualNews_i2t VisualNews_t2i WebQA
python hyperbolic/train.py \
  --model_name_or_path openai/clip-vit-large-patch14 \
  --bf16 \
  --lora \
  --subset_name CIRR \
  --num_sample_per_subset 100000 \
  --image_dir /home/thuy0050/ft49_scratch2/thuy0050/data/vlm2vec_train/MMEB-train \
  --max_len 77 \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps 10 \
  --learning_rate 3e-5 \
  --max_steps 2000 \
  --warmup_steps 200 \
  --save_steps 1000 \
  --normalize True \
  --temperature 0.02 \
  --per_device_train_batch_size 64 \
  --grad_cache True \
  --gc_q_chunk_size 32 \
  --gc_p_chunk_size 32 \
  --dataloader_num_workers 4 \
  --save_safetensors True \
  --remove_unused_columns False \
  --report_to none
