#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_eval3
#SBATCH --output=/scratch/mroycho1/GURU/qwen_eval3_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

echo "=== JOB START $(date) on $(hostname) ==="
nvidia-smi | head -5

module load ollama/0.12.10
echo "=== OLLAMA_HOST after module load: $OLLAMA_HOST ==="

# Start ollama — try both command names
ollama-start 2>/dev/null || ollama serve &
sleep 60

# Use the host set by the module, fallback to localhost
HOST="${OLLAMA_HOST:-http://localhost:11434}"
echo "=== Connecting to: $HOST ==="

# Wait for Ollama to be ready (up to 2 min)
for i in $(seq 1 12); do
    STATUS=$(curl -s --max-time 10 "$HOST/api/tags" | python3 -c "import json,sys; print('OK')" 2>/dev/null)
    if [ "$STATUS" = "OK" ]; then
        echo "=== Ollama ready after ${i}0 seconds ==="
        break
    fi
    echo "Waiting for Ollama... ($i/12)"
    sleep 10
done

# Test Qwen responds
RESP=$(curl -s --max-time 120 "$HOST/api/generate" \
  -d '{"model":"qwen2.5:72b","prompt":"say hi","stream":false,"options":{"num_predict":5}}')
echo "=== RAW TEST RESPONSE: ${RESP:0:200} ==="

RESP_TEXT=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('response','FAIL'))" 2>&1)
echo "=== PARSED: $RESP_TEXT ==="

if echo "$RESP_TEXT" | grep -qiE "FAIL|Error|500"; then
    echo "FATAL: Qwen not working on $HOST"
    exit 1
fi

echo "=== Qwen working! ==="

# Run corruption experiment
python plan_step31_three_experiments.py --part C \
    --host "$HOST" --model qwen2.5:72b --n 30
    
echo "=== Step 19 done at $(date) ==="
