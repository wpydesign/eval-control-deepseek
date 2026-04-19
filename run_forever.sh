#!/bin/bash
# Continuous evaluation loop. Run: bash run_forever.sh
# Processes ~6 prompts per 100s cycle. Resumes automatically.
cd "$(dirname "$0")"
while true; do
    COUNT=$(wc -l < logs/shadow_eval_live.jsonl 2>/dev/null || echo 0)
    PENDING=$(python3 -c "
import json,hashlib
ev=set()
try:
    with open('logs/shadow_eval_live.jsonl') as f:
        for l in f: ev.add(json.loads(l.strip()).get('query_id',''))
except: pass
c=0
try:
    with open('data/raw_prompts.jsonl') as f:
        for l in f:
            r=json.loads(l.strip()); p=r.get('prompt','').strip()
            if p and hashlib.sha256(p.encode()).hexdigest()[:12] not in ev: c+=1
except: pass
print(c)
" 2>/dev/null)
    echo "$(date -u) live=$COUNT pending=$PENDING"
    if [ "$PENDING" = "0" ] || [ -z "$PENDING" ]; then
        echo "ALL DONE"
        break
    fi
    timeout 120 python3 -u run_loop.py 2>&1 | tail -5
    sleep 2
done
