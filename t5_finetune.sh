#!/bin/bash
#SBATCH -N 1
#SBATCH -p shire-general
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/zhenwu/anlp-hw3/logs/t5_all_4.log
cd /home/zhenwu/anlp-hw3/KILT

for dataset in eli5; do
  python kilt/readers/t5/finetune.py \
    --data_dir=data/train \
    --dataset=$dataset \
    --model_name_or_path=t5-base \
    --learning_rate=1e-3 \
    --num_train_epochs=5 \
    --output_dir=/data/user_data/zhenwu/kilt/t5/$dataset \
    --n_gpu=4 \
    --do_train
done