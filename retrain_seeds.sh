#!/bin/bash
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --job-name=arc_retrain_v2
#SBATCH --output=/scratch/mroycho1/GURU/retrain_v2_%j.log

source /home/mroycho1/.bashrc
source activate mdebnath

cd /scratch/mroycho1/GURU
for SEED in 0 1 2 3 42 123; do
    echo "=== SEED $SEED ==="
    python3 plan_step3_guru.py \
        --n_episodes 5000 \
        --contrastive \
        --seed $SEED \
        --save checkpoints_planning/arc_final_seed${SEED}.pt \
        2>&1 | tail -10