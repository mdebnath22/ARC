#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_eval5
#SBATCH --output=/scratch/mroycho1/GURU/qwen_eval5_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

echo "=== JOB START $(date) on $(hostname) ==="

module load ollama/0.12.10
ollama serve &
sleep 120   # give model more time to load into GPU

HOST="${OLLAMA_HOST:-http://localhost:11434}"
[[ "$HOST" != http* ]] && HOST="http://$HOST"
echo "=== Using: $HOST ==="

# Wait up to 5 minutes for Ollama ready
for i in $(seq 1 30); do
    curl -sf "$HOST/api/tags" > /dev/null 2>&1 && echo "Ready at ${i}0s" && break
    sleep 10
done

# Test with 5 min timeout (model needs time for first inference)
RESP=$(curl -s --max-time 300 "$HOST/api/generate" \
  -d '{"model":"qwen2.5:72b","prompt":"say hi","stream":false,"options":{"num_predict":5}}')
echo "RAW: ${RESP:0:200}"

RESP_TEXT=$(echo "$RESP" | python3 -c \
  "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('response','FAIL'))" 2>&1)
echo "TEST: $RESP_TEXT"

if echo "$RESP_TEXT" | grep -qiE "^FAIL$|Error|500|canceled"; then
    echo "FATAL: Qwen failed"
    exit 1
fi

echo "=== Qwen working! Starting step 15 ==="
python qwen_eval.py --host "$HOST" --model qwen2.5:72b --overwrite
echo "=== Step 15 done at $(date) ==="

rm -f results_planning_ipc/qwen_ipc_eval.jsonl
python plan_step19_ipc_scale.py --phase eval \
    --model qwen2.5:72b --ollama_host "$HOST" --timeout 600
echo "=== Step 19 done at $(date) ==="
