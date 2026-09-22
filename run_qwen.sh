#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_eval
#SBATCH --output=/scratch/mroycho1/GURU/qwen_eval_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

echo "Job started on $(hostname) at $(date)"
module load ollama/0.12.10
ollama-start
sleep 60
echo "Ollama started, port: $OLLAMA_PORT"

# Test it works first
curl -s http://sc011:${OLLAMA_PORT}/api/generate \
  -d "{\"model\":\"qwen2.5:72b\",\"prompt\":\"say hi\",\"stream\":false,\"options\":{\"num_predict\":5}}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('TEST OK:', d.get('response','EMPTY')[:50])"

# Run step 15 (200 instances, ~3 hrs)
python qwen_eval.py \
    --host "http://sc011:${OLLAMA_PORT}" \
    --model qwen2.5:72b \
    --overwrite

echo "Step 15 done at $(date)"

# Then step 19 (1800 instances, remaining time)
rm -f results_planning_ipc/qwen_ipc_eval.jsonl
python plan_step19_ipc_scale.py --phase eval \
    --model qwen2.5:72b \
    --ollama_host "http://sc011:${OLLAMA_PORT}" \
    --timeout 600

echo "Step 19 done at $(date)"
