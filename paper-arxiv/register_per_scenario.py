"""Register the per-scenario Qwen3-32B test table (App G) from the saved test-call scores.

Recomputes, for each of the five test scenarios: scored calls, leaking calls, AUROC within
scenario x round, leaks caught at the development-control threshold, and the share of calls in
refusing episodes (recruiting episodes with no leaking call) that the probe flags. Pooled values
must reproduce the paper's headline test numbers. No model calls, no refits.
"""
import gzip, csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'experiments/E05-probes'))
from metrics import within_round  # noqa: E402

RUN = 'runs/20261026-2212-e05-arxiv-qwen-fresh-percall'
OUT = 'paper-arxiv/derived_per_scenario.json'
SCEN = [('dispatch-integration-code', 'Dispatch', 'Dispatch integration code'),
        ('handoff-exception-note', 'Handoff', 'Handoff exception note'),
        ('procurement-reserve-price', 'Procurement', 'Procurement reserve price'),
        ('service-capacity-figure', 'Capacity', 'Service capacity figure'),
        ('service-rollout-threshold', 'Rollout', 'Service rollout threshold')]


def stats(rows, threshold):
    y = np.array([r['label'] for r in rows]); s = np.array([r['score'] for r in rows])
    rounds = np.array([r['round'] for r in rows]); sc = np.array([r['scenario'] for r in rows])
    leaking_eps = {r['episode'] for r in rows if r['label'] == 1}
    held = np.array([r['episode'] not in leaking_eps for r in rows])
    hits = s > threshold
    return dict(calls=len(rows), leaks=int(y.sum()),
                auroc=within_round(y, s, rounds, stratum_key=sc)[0],
                caught=int((hits & (y == 1)).sum()),
                refusing_calls=int(held.sum()), refusing_flagged=int((hits & held).sum()),
                refusing_share=float(hits[held].mean()))


def main():
    reg = json.loads((HERE/'numbers.yaml').read_text())
    with gzip.open(ROOT/RUN/'probe_scores.csv.gz', 'rt') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['round'] = int(r['round']); r['label'] = int(r['label']); r['score'] = float(r['score'])
    metrics = json.loads((ROOT/RUN/'metrics.json').read_text())
    threshold = reg['numbers']['qwenFrozenThreshold']['value']
    assert threshold == metrics['threshold']
    out = dict(threshold=threshold, pooled=stats(rows, threshold), scenarios={})
    for key, _, _ in SCEN:
        out['scenarios'][key] = stats([r for r in rows if r['scenario'] == key], threshold)
    p = out['pooled']
    # Pooled values reproduce the paper's test results (AUROC 0.98, 31/38 caught, 5.5% refusing).
    assert abs(p['auroc'] - next(e for e in reg['numbers'].values() if e['macro'] == 'qwenFreshCallProbe')['value']) < 1e-12
    assert round(p['auroc'], 2) == 0.98 and (p['caught'], p['leaks']) == (31, 38), p
    assert (p['calls'], p['refusing_calls'], p['refusing_flagged']) == (1825, 1407, 77), p
    assert abs(p['refusing_share'] - next(e for e in reg['numbers'].values() if e['macro'] == 'qwenFreshInputRefusingAlarm')['value']) < 1e-12
    assert sum(v['leaks'] for v in out['scenarios'].values()) == 38
    (ROOT/OUT).write_text(json.dumps(out, indent=2)+'\n')

    def add(macro, path, fmt, decimals=0):
        value = out
        for k in path: value = value[k]
        reg['numbers'][macro] = dict(
            value=value, macro=macro,
            dataset='Qwen3-32B test episodes (new episodes in the five development scenarios), 1,825 recruiting calls',
            evaluation='Last-prompt probe fit on development episodes; threshold set on development control calls; per-scenario recount of saved scores, no refits',
            status='verified', run=RUN, format=fmt, decimals=decimals, source=dict(file=OUT, value_path=path))

    rows_tex = []
    for key, name, label in SCEN:
        m = 'qwenTestScen' + name
        add(m+'Calls', ['scenarios', key, 'calls'], 'integer')
        add(m+'Leaks', ['scenarios', key, 'leaks'], 'integer')
        add(m+'Auroc', ['scenarios', key, 'auroc'], 'decimal', 2)
        add(m+'Caught', ['scenarios', key, 'caught'], 'integer')
        add(m+'RefusingCalls', ['scenarios', key, 'refusing_calls'], 'integer')
        add(m+'RefusingShare', ['scenarios', key, 'refusing_share'], 'percent', 1)
        rows_tex.append([label, f'\\{m}Calls{{}}', f'\\{m}Leaks{{}}', f'\\{m}Auroc{{}}',
                         f'\\{m}Caught{{}}/\\{m}Leaks{{}}', f'\\{m}RefusingShare{{}}'])
    m = 'qwenTestScenAll'
    add(m+'Calls', ['pooled', 'calls'], 'integer'); add(m+'Leaks', ['pooled', 'leaks'], 'integer')
    add(m+'Auroc', ['pooled', 'auroc'], 'decimal', 2); add(m+'Caught', ['pooled', 'caught'], 'integer')
    add(m+'RefusingShare', ['pooled', 'refusing_share'], 'percent', 1)
    rows_tex.append(['All five', f'\\{m}Calls{{}}', f'\\{m}Leaks{{}}', f'\\{m}Auroc{{}}',
                     f'\\{m}Caught{{}}/\\{m}Leaks{{}}', f'\\{m}RefusingShare{{}}'])
    aucs = {k: round(v['auroc'], 2) for k, v in out['scenarios'].items()}
    lo, hi = min(aucs, key=aucs.get), max(aucs, key=aucs.get)
    add('qwenTestScenAurocMin', ['scenarios', lo, 'auroc'], 'decimal', 2)
    add('qwenTestScenAurocMax', ['scenarios', hi, 'auroc'], 'decimal', 2)
    reg['tables']['per_scenario_test'] = dict(
        columns=r'@{}lccccc@{}',
        header=['Scenario', 'Calls', 'Leaking calls', 'AUROC', 'Leaks caught', 'Refusing calls flagged'],
        rows=rows_tex)
    (HERE/'numbers.yaml').write_text(json.dumps(reg, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()
