#!/bin/bash
#SBATCH -N 1
#SBATCH -p shire-general
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/zhenwu/anlp-hw3/logs/t5/generate_2.log
cd /home/zhenwu/anlp-hw3/KILT

for dataset in fever trex zeroshot nq hotpotqa triviaqa eli5 wow; do
  echo "===========Generating $dataset==========="
  for (( k=0; k<=20; k++ )); do
    echo "Generating top $k"
    python kilt/readers/t5/evaluate_kilt_task.py t5-base data/dev/${dataset}-dev-kilt.txt /data/user_data/zhenwu/kilt/t5/${dataset}/${dataset}-dev-kilt-$k-prefix.txt /data/user_data/zhenwu/kilt/t5/${dataset} retrieval_results/dpr/${dataset}-dev-kilt.jsonl $k
  done
done