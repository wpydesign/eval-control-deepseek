#!/bin/bash
# Run 300-sample evaluation in chunks of 5.
# Each chunk is a separate process — survives kills.
# Resume-safe: skips already-evaluated prompts.
set -e

cd /home/z/my-project/eval-control-repo
export DEEPSEEK_API_KEY="sk-f55cb3459edd4becb8d6f83db3afd6d1"

CHUNK=5
TOTAL=300

for start in $(seq 0 $CHUNK $((TOTAL-1))); do
    end=$((start + CHUNK))
    if [ $end -gt $TOTAL ]; then end=$TOTAL; fi
    
    echo "=== Running chunk: $start to $end ==="
    python3 -u eval_batch.py --start $start --end $end --perturb 2 --contexts 3
    
    count=$(wc -l < calibration_dataset_300.jsonl 2>/dev/null || echo 0)
    echo "=== Progress: $count / $TOTAL ==="
    echo ""
    
    # Small pause between chunks
    sleep 2
done

echo "=== ALL DONE ==="
wc -l calibration_dataset_300.jsonl
