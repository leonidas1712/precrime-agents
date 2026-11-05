"""E17: pack the GPU replay payload. CPU only, no API, no GPU.

One forward per Test 1 / Test 2 row: the target request that made the scored write
(PF forbidden, PF compromise, PA allowed). Activations are read at its last prompt token.

    .venv/bin/python experiments/E17-crossed/replay_payload.py runs/<stage 1> runs/<stage 2> --out private/crossed/e17_replay
    .venv/bin/python experiments/E17-crossed/replay_payload.py ... --tokens   # also count prompt tokens with the local Qwen3-32B tokenizer

Writes <out>.payload.json (call + request + record, the format E15's modal_app reads) and <out>.calls.json (labels and design variables).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'experiments/E02-activation-replay')]
from crossed import score_crossed  # noqa: E402
from main_report import setup_feats  # noqa: E402

ROWS = {('PF', 'forbidden'): 'test1_forbidden', ('PF', 'allowed'): 'test1_compromise', ('PA', 'allowed'): 'test2_pa_allowed'}


def scored_request(ep, call_id):
    """The completed target request whose response contains the scored write's tool call."""
    for i, r in enumerate(ep.get('requests', [])):
        if r.get('actor') != 'target' or not r.get('response') or r.get('error'):
            continue
        msg = ((r['response'].get('choices') or [{}])[0]).get('message') or {}
        if any(c.get('id') == call_id for c in msg.get('tool_calls') or []):
            return i, r
    raise LookupError(call_id)


def build(run_dirs):
    payload, calls = [], []
    for d in run_dirs:
        for path in sorted((Path(d) / 'episodes').glob('*.json')):
            if '.attempt' in path.name or path.name.endswith('.partial.json'):
                continue
            raw = path.read_bytes()
            ep = json.loads(raw)
            m = ep['scenario_spec']['metadata']
            lab = score_crossed(ep)
            if not lab.get('in_label_B'):
                continue
            row = ROWS.get((m['arm'], lab['label_B']))
            if row is None:
                continue
            first = lab['first_write_after_request']
            idx, rec = scored_request(ep, first['call_id'])
            call = {'call_id': f'{Path(d).name}/{path.name}::request_idx={idx}', 'episode': f'{Path(d).name}/{path.name}',
                    'episode_sha256': hashlib.sha256(raw).hexdigest(), 'episode_id': m['episode_id'],
                    'scenario': m['spray_row_id'], 'cell': m['arm'], 'condition': row, 'row': row,
                    'label_forbidden': int(lab['label_B'] == 'forbidden'), 'round': rec.get('round'), 'request_idx': idx,
                    'setup': setup_feats(m)}
            payload.append({'call': call, 'request': rec['request'], 'record': rec})
            calls.append(call)
    return payload, calls


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--tokens', action='store_true')
    a = p.parse_args()
    payload, calls = build(a.runs)
    Path(a.out + '.payload.json').write_text(json.dumps(payload, ensure_ascii=False))
    Path(a.out + '.calls.json').write_text(json.dumps(calls, indent=1))
    out = {'forwards': len(payload), 'rows': Counter(c['row'] for c in calls),
           'by_scenario': Counter(f"{c['scenario']}|{c['row']}" for c in calls)}
    if a.tokens:
        from transformers import AutoTokenizer
        from replay import render
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-32B', revision='9216db5781bf21249d130ec9da846c4624c16137',
                                            local_files_only=True)
        n = [len(render(tok, item['request'], 'from-request')[1]) for item in payload]
        thinking = Counter(render(tok, item['request'], 'from-request')[3] for item in payload[:20])
        out['prompt_tokens'] = {'total': sum(n), 'mean': sum(n) / len(n), 'max': max(n), 'min': min(n)}
        out['thinking_in_first_20'] = thinking
    print(json.dumps(out, indent=1, default=dict))
