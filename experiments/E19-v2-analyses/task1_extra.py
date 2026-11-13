"""Task 1 extra: what is the latest message on E15 sharing calls that do not follow a read (by arm)?"""
import json, re
from collections import Counter, defaultdict
from common_v2 import OUT, ROOT, request_for
calls = json.loads((ROOT / 'runs/20261028-1039-e15-giving-in-scaleup/calls.json').read_text())
HEDGE = re.compile(json.loads((ROOT / 'experiments/E15-giving-in-scaleup/plan.json').read_text())['hedge_regex'])
out = defaultdict(Counter)
for c in calls:
    if not c['label'] or c['follows_read'] or (c['arm'] == 'LATE' and c['before_ask']): continue
    g = 'FORBID' if c['arm'] in ('FORBID', 'FORBID_OLD') else c['arm']
    req, hp = request_for(c['call_id']); last = req['messages'][-1]; txt = json.dumps(last, ensure_ascii=False)
    kind = 'peer message' if last.get('role') == 'user' else ('delivered confirmation' if 'delivered' in txt else 'other tool result')
    out[g][kind] += 1; out[g]['n'] += 1
    out[g]['peer asks for format / partial / confirm'] += bool(re.search(r'(?i)\b(format|at least|confirm|partial|first (few|\d+) (characters|digits))\b', txt)) and last.get('role') == 'user'
    out[g]['hedge regex in latest message'] += bool(HEDGE.search(txt))
    out[g]['round>=3'] += c['round'] >= 3
res = {g: dict(v) for g, v in out.items()}
(OUT / 'task1_extra.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
