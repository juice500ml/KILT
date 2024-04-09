#!/bin/bash
#SBATCH -N 1
#SBATCH -p shire-general
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/zhenwu/anlp-hw3/logs/t5/eval/eval_prefix.log
cd /home/zhenwu/anlp-hw3/KILT

for dataset in fever trex zeroshot nq hotpotqa triviaqa eli5 wow; do
  echo "===========Evaluating $dataset==========="
  for (( k=0; k<=20; k++ )); do
    echo "Evaluating top $k"
    python kilt/readers/t5/custom_eval.py --gold_answers data/dev/$dataset-dev-kilt_answers.txt --predictions /data/user_data/zhenwu/kilt/t5/$dataset/$dataset-dev-kilt-$k-prefix.txt --dataset $dataset --k $k --mode eval
  done
done