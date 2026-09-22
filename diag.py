# diagnose_eto.py  — run once, paste the output here
import json
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="agent-eto/eto-sft-trajectory",
    filename="data/sciworld_sft.json",
    repo_type="dataset",
)

with open(local_path) as f:
    raw = json.load(f)

print(f"Total records: {len(raw)}")
print(f"Keys in record 0: {list(raw[0].keys())}")
print()

# Print full first record
ex = raw[0]
print("=== RECORD 0 ===")
print(f"id: {ex.get('id')}")
convs = ex.get("conversations", [])
print(f"num turns: {len(convs)}")
for i, turn in enumerate(convs[:6]):
    role = turn.get("from", turn.get("role", "?"))
    val  = turn.get("value", turn.get("content", ""))
    print(f"\n  turn[{i}] from={role}")
    print(f"  value[:400] = {repr(val[:400])}")

print()
print("=== RECORD 1 ===")
ex1 = raw[1]
convs1 = ex1.get("conversations", [])
for i, turn in enumerate(convs1[:3]):
    role = turn.get("from", "?")
    val  = turn.get("value", "")
    print(f"\n  turn[{i}] from={role}")
    print(f"  value[:300] = {repr(val[:300])}")

# Check last turn of record 0 — often contains score/result
print("\n=== LAST 2 TURNS of record 0 ===")
for turn in convs[-2:]:
    role = turn.get("from", "?")
    val  = turn.get("value", "")
    print(f"\n  from={role}")
    print(f"  value[:400] = {repr(val[:400])}")