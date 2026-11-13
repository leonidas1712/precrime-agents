"""Task 3 (queue item 2): gpt-oss read flag on the one-call-earlier ('strict1') label, paper metric.

Rows, labels, folds: exactly the rows behind \\gptossStrictNextProbe (0.62): run
20261026-1221-e05-lastprompt-percall-6013, cohort=matched, label_variant=strict1, analysis=loso (3 seeds).
Read flags (saved, no refit):
  - own_protected_read_call: the call's own reply calls read_file on the protected file
    (runs/20261026-0526-e05-reasoning-stream/strict_future_readstep.json 'details'; it sees the call's output,
    so it is not information-matched to a last-prompt probe);
  - latest_incoming_protected_read / previous_call_protected_read (input-side flags, from immediate_readstep.json definitions,
    recomputed here from the saved episodes).
Metric: stream_metrics (within scenario x round AUROC, family bootstrap 1000, seed 20261025), paired difference vs the probe.

    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task3_readflag_next.py
"""
import csv, gzip, json
from collections import defaultdict
import numpy as np
from common_v2 import *
from common_v2 import OUT, load_episode, split_id

OLD = Path('/Users/leo/collusion-research')
SRC = OLD / 'runs/20261026-1221-e05-lastprompt-percall-6013/probe_scores.csv.gz'
READ = OLD / 'runs/20261026-0526-e05-reasoning-stream/strict_future_readstep.json'
META = OLD / 'runs/20261026-0522-e05-reasoning-stream-data/replies.json'
SAVED_METRICS = OLD / 'runs/20261026-1221-e05-lastprompt-percall-6013/metrics.json'


def main():
    meta = {r['row_id']: r for r in json.loads(META.read_text())}
    by = defaultdict(dict)
    for r in csv.DictReader(gzip.open(SRC, 'rt')):
        if r['cohort'] == 'matched' and r['label_variant'] == 'strict1' and r['analysis'] == 'loso':
            by[r['row_id']][int(r['seed'])] = r
    ids = sorted(by); assert all(len(by[i]) == 3 for i in ids)
    rows = [dict(meta[i], row_id=i) for i in ids]
    for r, i in zip(rows, ids):
        r['scenario'] = by[i][42]['scenario']; r['round'] = int(by[i][42]['round'])
    y = np.array([int(by[i][42]['label']) for i in ids])
    probe = np.array([[float(by[i][s]['score']) for i in ids] for s in (42, 43, 44)])
    folds = np.array([[int(by[i][s]['fold']) for i in ids] for s in (42, 43, 44)])
    det = {d['row_id']: d for d in json.loads(READ.read_text())['details']}
    assert set(ids) == set(det), (len(set(ids) - set(det)), len(set(det) - set(ids)))
    own = np.array([det[i]['protected_read_call'] for i in ids], float)
    malformed = np.array([det[i]['malformed_read_call'] for i in ids], float)
    # input-side flags
    latest_read = []; prev_read = []
    for i in ids:
        ep, idx = split_id(i); e = load_episode(ep)
        hp = e['scenario_spec']['honeypots']
        msgs = e['requests'][idx]['request']['messages']
        last = msgs[-1]
        latest_read.append(last.get('role') == 'tool' and any(v in json.dumps(last, ensure_ascii=False) for v in hp))
        prev = e['requests'][idx - 1] if idx > 0 else None
        pr = False
        if prev is not None:
            protected = {p for p, c in e['scenario_spec']['files'].items() if any(v in c for v in hp)}
            msg = ((prev.get('response') or {}).get('choices') or [{}])[0].get('message') or {}
            for c in msg.get('tool_calls') or []:
                fn = c.get('function') or {}; a = fn.get('arguments') or {}
                if isinstance(a, str):
                    try: a = json.loads(a)
                    except json.JSONDecodeError: continue
                if isinstance(a, dict) and fn.get('name') == 'read_file' and a.get('path') in protected: pr = True
        prev_read.append(pr)
    latest_read = np.array(latest_read, float); prev_read = np.array(prev_read, float)
    allids = np.arange(len(ids))
    mp, bp = stream_metrics(rows, allids, y, probe, folds, 1000)
    res = dict(rows=len(ids), positives=int(y.sum()), families=len({r['family'] for r in rows}), episodes=mp['episodes'],
               probe=dict(auc=mp['within_round_auc'], ci95=mp['within_round_ci95']),
               paper_check=dict(recomputed=mp['within_round_auc'], saved=json.loads(SAVED_METRICS.read_text())['matched/strict1/loso']['within_round_auc']))
    for name, flag in [('own_protected_read_call', own), ('latest_input_is_protected_read', latest_read), ('previous_call_read_protected_file', prev_read)]:
        m, b = stream_metrics(rows, allids, y, np.tile(flag, (3, 1)), folds, 1000)
        d = bp['within_round'] - b['within_round']
        res[name] = dict(auc=m['within_round_auc'], ci95=m['within_round_ci95'],
                         positives_flagged=int(flag[y == 1].sum()), negatives_flagged=int(flag[y == 0].sum()),
                         probe_minus_flag=dict(estimate=mp['within_round_auc'] - m['within_round_auc'], ci95=np.nanquantile(d, [.025, .975]).tolist()))
    res['malformed_read_calls_among_positives'] = int(malformed[y == 1].sum())
    # reference: reasoning probe (token max) on the same label, from the saved strict_future run
    sf = json.loads((OLD / 'runs/20261026-0526-e05-reasoning-stream/strict_future.json').read_text())['results']['h1/all/loso/token_max']
    res['reference_reasoning_probe_token_max_strict1'] = dict(auc=sf['within_round_auc'], ci95=sf['within_round_ci95'], rows=sf['rows'], positives=sf['positives'],
                                                              source='runs/20261026-0526-e05-reasoning-stream/strict_future.json h1/all/loso/token_max')
    (OUT / 'task3_readflag_next.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
