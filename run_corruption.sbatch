#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J fgrag-all
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --export=NONE

source /home/mroycho1/.bashrc
conda activate mdebnath
module load ollama/0.12.10
ollama-start
sleep 60  # wait for model to load

HOST="http://$(hostname):11434"
echo "Host: $HOST"

# Verify it works
python3 -c "
import urllib.request, json, socket
host = f'http://{socket.gethostname()}:11434'
tags = json.loads(urllib.request.urlopen(f'{host}/api/tags', timeout=10).read())
print('Models:', [m['name'] for m in tags['models']])
payload = json.dumps({'model':'qwen2.5:72b','prompt':'Say: (pick-up b1)',
    'stream':False,'options':{'num_predict':10,'temperature':0.0}}).encode()
req = urllib.request.Request(f'{host}/api/generate', data=payload,
      headers={'Content-Type':'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=120) as r:
    resp = json.loads(r.read())
    print('Test OK:', resp.get('response','')[:30])
"

# Run corruption experiment
python plan_step31_three_experiments.py --part C \
    --host "$HOST" --model qwen2.5:72b --n 30

