#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=llm_ablation
#SBATCH --output=/scratch/mroycho1/GURU/llm_ablation_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

echo "=== START $(date) on $(hostname) ==="
module load ollama/0.12.10
ollama serve &
sleep 90

HOST="${OLLAMA_HOST:-http://localhost:11434}"
[[ "$HOST" != http* ]] && HOST="http://$HOST"
echo "=== Host: $HOST ==="

# Quick test
RESP=$(curl -s --max-time 60 "$HOST/api/generate" \
  -d '{"model":"glm4:latest","prompt":"say hi","stream":false,"options":{"num_predict":3}}')
echo "=== Test: ${RESP:0:80} ==="

# Run LLM ablation (glm4 first — small, fast)
python plan_step23_v2.py --part llms \
    --host "$HOST" \
    --n 50 \
    --timeout 15

echo "=== DONE $(date) ==="
