#!/bin/bash
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=corruption_exp
#SBATCH --output=/scratch/mroycho1/GURU/corruption_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

echo "=== JOB START $(date) on $(hostname) ==="
nvidia-smi | grep -E "GPU Name|Memory"

module load ollama/0.12.10
ollama serve &
sleep 90

HOST="${OLLAMA_HOST:-http://localhost:11434}"
[[ "$HOST" != http* ]] && HOST="http://$HOST"
echo "=== Using: $HOST ==="

for i in $(seq 1 18); do
    curl -sf "$HOST/api/tags" > /dev/null 2>&1 && echo "Ready at ${i}0s" && break
    sleep 10
done

RESP=$(curl -s --max-time 120 "$HOST/api/generate" \
  -d '{"model":"qwen2.5:72b","prompt":"say hi","stream":false,"options":{"num_predict":5}}')
RESP_TEXT=$(echo "$RESP" | python3 -c \
  "import json,sys; print(json.load(sys.stdin).get('response','FAIL'))" 2>&1)
echo "TEST: $RESP_TEXT"

if echo "$RESP_TEXT" | grep -qiE "FAIL|Error|500"; then
    echo "FATAL: Qwen 72B needs 80GB VRAM — checking GPU memory..."
    nvidia-smi
    # Try smaller model as fallback
    RESP2=$(curl -s --max-time 120 "$HOST/api/generate" \
      -d '{"model":"qwen2.5:7b","prompt":"say hi","stream":false,"options":{"num_predict":5}}')
    echo "7B fallback: $RESP2"
    exit 1
fi

echo "=== Qwen working — running corruption experiment ==="

python3 plan_step31_three_experiments.py \
    --part C \
    --host "$HOST" \
    --model qwen2.5:72b \
    --n 40

echo "=== Done at $(date) ==="
python3 -c "
import json
from pathlib import Path
f = Path('results_planning/progressive_corruption.json')
if not f.exists():
    print('ERROR: output file not found')
    exit(1)
r = json.loads(f.read_text())
print('Level     LLM%    ARC|rho|   |O||rho|')
print('-'*45)
for level, v in sorted(r.items(), key=lambda x: int(x[0].replace('%',''))):
    llm = v.get('llm_success', v.get('llm_rate', 0)) * 100
    arc = abs(v.get('rho_arc', v.get('arc_rho', 0)))
    obj = abs(v.get('rho_obj', v.get('obj_rho', 0)))
    print(f'  {level:>5}   {llm:5.1f}%   {arc:.3f}      {obj:.3f}')
"
