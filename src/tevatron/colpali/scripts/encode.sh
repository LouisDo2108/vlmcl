#!/bin/bash

## FOR PARTITION GPU, A100
##SBATCH --partition=gpu
##SBATCH --gres=gpu:A100:1
##SBATCH --nodelist=m3n100,m3n101,m3n102,m3n103,m3n104,m3n105,m3n106,m3n107,m3n108,m3n109,m3n110,m3n111,m3n112

## FOR PARTITION GPU, L40S
## SBATCH --partition=gpu
## SBATCH --gres=gpu:L40S:1

## FOR FIT PARTITION
#SBATCH --partition=fit
#SBATCH --nodelist=m3u000,m3u001,m3u002,m3u003,m3u004,m3u005,m3u006,m3u007,m3u008
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq

#SBATCH --account=mg61
#SBATCH --job-name=thuy0050
#SBATCH --output=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/logs/temporal/slurm-%x-%j.out
#SBATCH --error=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/logs/temporal/slurm-%x-%j.err
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

#SBATCH --mail-user=tuan.huynh1@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ==== ENVIRONMENT SETUP ====
source ~/.bashrc
mamba activate colpali

export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/vlmcl

BASE_MODEL_NAME_OR_PATH="/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/models/colqwen3-base"
MODEL_NAME_OR_PATH="/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/colqwen3_tevatron/1epoch_docvqa_inbatchneg_after_arxiv_qa"
LORAS_NAME_OR_PATH=$MODEL_NAME_OR_PATH/lora_paths.txt # $MODEL_NAME_OR_PATH/checkpoint-311
DATASET_NAME="vidore/docvqa_test_subsampled_beir"
QREL_PATH="/home/thuy0050/mg61_scratch2/thuy0050/data/vlmcl/vidore_v1_qrels/docvqa_test_subsampled_beir_qrel.txt"

echo $MODEL_NAME_OR_PATH

# "vidore/arxivqa_test_subsampled_beir"
# "vidore/docvqa_test_subsampled_beir"
# "vidore/infovqa_test_subsampled_beir"
# "vidore/tatdqa_test_beir"


OUTPUT_DIR=$MODEL_NAME_OR_PATH/out/$DATASET_NAME
mkdir -p $OUTPUT_DIR


python src/tevatron/colpali/encode.py \
  --output_dir=$OUTPUT_DIR \
  --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
  --lora_name_or_path $LORAS_NAME_OR_PATH \
  --bf16 \
  --dataset_name $DATASET_NAME \
  --dataset_split test \
  --dataset_config corpus \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 4 \
  --embedding_projection True \
  --encode_output_path $OUTPUT_DIR/corpus.pkl

python src/tevatron/colpali/encode.py \
  --output_dir=$OUTPUT_DIR \
  --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
  --lora_name_or_path $LORAS_NAME_OR_PATH \
  --bf16 \
  --dataset_name $DATASET_NAME \
  --dataset_split test \
  --dataset_config queries \
  --query_max_len 256 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 4 \
  --embedding_projection True \
  --encode_output_path $OUTPUT_DIR/query.pkl \
  --encode_is_query

python src/tevatron/colpali/search.py \
    --query_reps $OUTPUT_DIR/query.pkl \
    --passage_reps $OUTPUT_DIR/corpus.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $OUTPUT_DIR/rank.txt

# --remove_query: there is a check in convert_result_to_trec that check: "if args.remove_query and qid == docid: continue", so we will remove it
python -m tevatron.utils.format.convert_result_to_trec \
  --input $OUTPUT_DIR/rank.txt \
  --output $OUTPUT_DIR/rank.trec

python -m pyserini.eval.trec_eval -c \
  -m recall.10,100 -m ndcg_cut.10 -M 100 \
  $QREL_PATH \
  $OUTPUT_DIR/rank.trec > $OUTPUT_DIR/out.txt 