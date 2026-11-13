"""Robustness of the headline per-call AUROCs to the metric, from saved scores only.

No refits, no API, no GPU. For each detector x set this reproduces the paper's
within scenario x round AUROC (positive-weighted; single-class groups dropped) and its
family-bootstrap 95% interval (1,000 draws, seed 20261025), then reports:
  - within scenario only (rounds pooled inside each scenario),
  - pooled AUROC over all calls,
  - what the within-group metric keeps (calls, negatives, groups, positive x negative pairs),
  - leave-one-group-out min/max and per-scenario AUROC,
  - per-seed AUROC where the scores have three split seeds,
  - paired bootstrap differences probe minus comparator under each metric.

    uv run --no-project --with numpy --with scikit-learn python experiments/E18-robustness/metric_robustness.py
"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'experiments/E05-probes'))
from collusion.runs import start_run, write_json_atomic  # noqa: E402
from evaluate import ranking_summary  # noqa: E402
from metrics import within_round  # noqa: E402

CR = Path('/Users/leo/collusion-research/runs'); PA = ROOT/'runs'
REPS, SEED = 1000, 20261025
SRC = dict(
    qwen_dev_text=PA/'20261026-2245-e05-arxiv-qwen-text-workflow/scores.csv.gz',
    qwen_jev=CR/'20261026-1309-e06-percall-input-jev-qwen/probe_scores.csv',
    gpt_dev_text=PA/'20261026-2242-e05-arxiv-gpt-text-workflow/scores.csv.gz',
    gpt_dev_lastprompt=CR/'20261026-1221-e05-lastprompt-percall-6013/probe_scores.csv.gz',
    gpt_dev_jev=CR/'20261026-1319-e06-percall-input-jev-l4/probe_scores.csv',
    qwen_test=PA/'20261026-2212-e05-arxiv-qwen-fresh-percall/probe_scores.csv.gz',
    qwen_test_bow=PA/'20261028-0228-e12-bow-test-auroc/bow_scores.csv.gz',
    gpt_test=PA/'20261026-2222-e05-arxiv-gpt-fresh-percall/probe_scores.csv.gz',
    gpt_test_stream=CR/'20261026-1302-e05-stream-confirm-score/probe_scores.csv')
# Saved metrics that each recomputed headline must match exactly (path into metrics.json), plus the paper value.
SAVED = {
    ('qwen_dev', 'probe'): (PA/'20261026-2211-e05-arxiv-qwen-controls', ['metrics', 'probe'], .960),
    ('qwen_dev', 'jev'): (PA/'20261026-2211-e05-arxiv-qwen-controls', ['metrics', 'jev'], .758),
    ('qwen_dev', 'text_latest'): (PA/'20261026-2245-e05-arxiv-qwen-text-workflow', ['metrics', 'latest_input'], .925),
    ('gpt_dev', 'probe'): (PA/'20261026-1221-e05-lastprompt-percall-6013', ['matched/immediate/loso'], .906),
    ('gpt_dev', 'reasoning'): (PA/'20261026-1221-e05-lastprompt-percall-6013', ['matched/immediate/loso', 'reasoning'], .933),
    ('gpt_dev', 'jev'): (PA/'20261026-1319-e06-percall-input-jev-l4', ['loso'], .397),
    ('gpt_dev', 'text_full'): (PA/'20261026-2242-e05-arxiv-gpt-text-workflow', ['metrics', 'full_input'], .863),
    ('qwen_test', 'probe'): (PA/'20261026-2212-e05-arxiv-qwen-fresh-percall', [], .976),
    ('qwen_test', 'jev'): (PA/'20261026-2224-e05-arxiv-fresh-comparisons', ['qwen', 'comparator'], .749),
    ('qwen_test', 'text_bow'): (PA/'20261028-0228-e12-bow-test-auroc', ['bow'], .957),
    ('gpt_test', 'probe'): (PA/'20261026-2222-e05-arxiv-gpt-fresh-percall', [], .818),
    ('gpt_test', 'reasoning'): (PA/'20261026-2224-e05-arxiv-fresh-comparisons', ['gpt_oss', 'comparator'], .905),
}
PAIRS = {'qwen_dev': [('probe', 'jev'), ('probe', 'text_latest')],
         'gpt_dev': [('probe', 'jev'), ('reasoning', 'jev'), ('probe', 'text_full'), ('reasoning', 'text_full'), ('reasoning', 'probe')],
         'qwen_test': [('probe', 'jev'), ('probe', 'text_bow')],
         'gpt_test': [('reasoning', 'probe')]}


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f: return list(csv.DictReader(f))


def one_score(rows, key='score', analysis=None):
    """row_id -> score, asserting a single value per call across seeds/analyses."""
    out = {}
    for r in rows:
        if analysis and r['analysis'] != analysis: continue
        v = float(r[key]); assert out.setdefault(r['row_id'], v) == v, r['row_id']
    return out


def seeded(rows, ids, col, filt=lambda r: True):
    pos = {k: i for i, k in enumerate(ids)}; m = np.full((3, len(ids)), np.nan)
    for r in rows:
        if filt(r) and r['row_id'] in pos: m[int(r['seed'])-42, pos[r['row_id']]] = float(r[col])
    assert np.isfinite(m).all(); return m


def load_sets():
    sets = {}
    for name, key in [('qwen_dev', 'qwen_dev_text'), ('gpt_dev', 'gpt_dev_text')]:
        rows = read(SRC[key]); base = {r['row_id']: r for r in rows if r['seed'] == '42'}; ids = sorted(base)
        d = dict(ids=ids, y=np.array([int(base[k]['label']) for k in ids]), rounds=np.array([int(base[k]['round']) for k in ids]),
                 scen=np.array([base[k]['scenario'] for k in ids]), fam=np.array([base[k]['family'] for k in ids]), det={})
        d['det']['probe'] = seeded(rows, ids, 'last_prompt')
        text = 'latest_input' if name == 'qwen_dev' else 'full_input'
        t = seeded(rows, ids, text); assert (t == t[0]).all(); d['det']['text_'+text.split('_')[0]] = t[:1]
        sets[name] = d
    q = sets['qwen_dev']; jev = one_score(read(SRC['qwen_jev'])); q['det']['jev'] = np.array([[jev[k] for k in q['ids']]])
    g = sets['gpt_dev']; lp = read(SRC['gpt_dev_lastprompt'])
    sel = lambda r: r['cohort'] == 'matched' and r['label_variant'] == 'immediate' and r['analysis'] == 'loso'
    assert np.array_equal(seeded(lp, g['ids'], 'score', sel), g['det']['probe'])
    g['det']['reasoning'] = seeded(lp, g['ids'], 'reasoning_score', sel)
    jev = one_score(read(SRC['gpt_dev_jev']), analysis='loso'); g['det']['jev'] = np.array([[jev[k] for k in g['ids']]])
    for name, key in [('qwen_test', 'qwen_test'), ('gpt_test', 'gpt_test')]:
        rows = read(SRC[key]); ids = [r['row_id'] for r in rows]; assert len(set(ids)) == len(ids)
        sets[name] = dict(ids=ids, y=np.array([int(r['label']) for r in rows]), rounds=np.array([int(r['round']) for r in rows]),
                          scen=np.array([r['scenario'] for r in rows]), fam=np.array([r['family'] for r in rows]),
                          det={'probe': np.array([[float(r['score']) for r in rows]])})
    q = sets['qwen_test']; jev = one_score(read(SRC['qwen_jev'])); q['det']['jev'] = np.array([[jev[k] for k in q['ids']]])
    bow = {r['row_id']: r for r in read(SRC['qwen_test_bow'])}
    assert all(float(bow[k]['probe']) == s for k, s in zip(q['ids'], q['det']['probe'][0]))
    q['det']['text_bow'] = np.array([[float(bow[k]['bow']) for k in q['ids']]])
    g = sets['gpt_test']; st = {r['row_id']: r for r in read(SRC['gpt_test_stream'])}
    g['det']['reasoning'] = np.array([[float(st[k]['max_score']) for k in g['ids']]])
    return sets


def coverage(y, keys):
    groups = defaultdict(list)
    for i, k in enumerate(keys): groups[k].append(i)
    kept = [ix for ix in groups.values() if 0 < y[ix].sum() < len(ix)]
    ix = np.concatenate(kept) if kept else np.array([], int)
    return dict(calls=len(y), negatives=int((y == 0).sum()), positives=int(y.sum()), groups=len(groups),
                kept_calls=len(ix), kept_negatives=int((y[ix] == 0).sum()), kept_positives=int(y[ix].sum()), kept_groups=len(kept),
                pairs=int(sum(y[g].sum()*(len(g)-y[g].sum()) for g in kept)))


def summarize(d, S, rounds, key):
    res, wb, pb = ranking_summary(np.tile(d['y'], (len(S), 1)), S, rounds, d['fam'], REPS, seed=SEED, stratum_key=key)
    return res, wb, pb


def analyse(name, d):
    y, n = d['y'], len(d['y']); zeros = np.zeros(n, int)
    cov = dict(scenario_round=coverage(y, list(zip(d['scen'], d['rounds']))), scenario=coverage(y, list(d['scen'])), pooled=coverage(y, [0]*n))
    out = dict(coverage=cov, detectors={}); boots = {}
    for det, S in d['det'].items():
        h, hwb, hpb = summarize(d, S, d['rounds'], d['scen'])
        s, swb, _ = summarize(d, S, zeros, d['scen'])
        boots[det] = dict(scenario_round=hwb, scenario=swb, pooled=hpb)
        r = dict(scenario_round=dict(auc=h['within_round_auc'], ci95=h['within_round_ci95'], by_seed=h['within_round_by_seed']),
                 scenario=dict(auc=s['within_round_auc'], ci95=s['within_round_ci95'], by_seed=s['within_round_by_seed']),
                 pooled=dict(auc=h['pooled_auc'], ci95=h['pooled_ci95'], by_seed=h['pooled_by_seed']), n_seeds=len(S))
        # leave one kept scenario x round group out (point estimate, seed mean)
        gk = np.array([f'{c}|{t}' for c, t in zip(d['scen'], d['rounds'])]); logo = {}
        for g in sorted(set(gk)):
            m = gk == g
            if not (0 < y[m].sum() < m.sum()): continue
            logo[g] = float(np.mean([within_round(y[~m], s_[~m], d['rounds'][~m], stratum_key=d['scen'][~m])[0] for s_ in S]))
        lo, hi = min(logo, key=logo.get), max(logo, key=logo.get)
        r['leave_one_group_out'] = dict(min=logo[lo], min_group=lo, max=logo[hi], max_group=hi, n=len(logo), values=logo)
        per = {}
        for c in sorted(set(d['scen'])):
            m = d['scen'] == c; sub = dict(y=y[m], fam=d['fam'][m])
            calls = int(m.sum()); leaks = int(y[m].sum())
            if leaks == 0 or leaks == calls: per[c] = dict(calls=calls, leaks=leaks, auc=None); continue
            a, _, _ = summarize(sub, S[:, m], d['rounds'][m], None)
            per[c] = dict(calls=calls, leaks=leaks, kept=coverage(y[m], list(d['rounds'][m])), auc=a['within_round_auc'], ci95=a['within_round_ci95'],
                          pooled_auc=a['pooled_auc'], pooled_ci95=a['pooled_ci95'])
        r['per_scenario'] = per
        out['detectors'][det] = r
    out['paired'] = {}
    for a, b in PAIRS[name]:
        out['paired'][f'{a}_minus_{b}'] = {v: dict(estimate=out['detectors'][a][v]['auc']-out['detectors'][b][v]['auc'],
                                                    ci95=np.nanquantile(boots[a][v]-boots[b][v], [.025, .975]).tolist())
                                           for v in ['scenario_round', 'scenario', 'pooled']}
    return out


def check(name, det, r):
    path, keys, paper = SAVED[(name, det)]; m = json.loads((path/'metrics.json').read_text())
    for k in keys: m = m[k]
    h = r['scenario_round']
    ok = dict(auc=abs(h['auc']-m['within_round_auc']) < 1e-12, ci=np.allclose(h['ci95'], m['within_round_ci95'], atol=1e-12, rtol=0),
              pooled=abs(r['pooled']['auc']-m['pooled_auc']) < 1e-12, paper_match=round(h['auc'], 3) == paper)
    return dict(saved_run=path.name, saved_auc=m['within_round_auc'], saved_ci95=m['within_round_ci95'], paper=paper, **ok)


def main():
    run = start_run('e18-metric-robustness', experiment='E18', tags=['cpu-only', 'saved-scores-only'],
                    parents=sorted({p.name for p, _, _ in SAVED.values()} | {p.parent.name for p in SRC.values()}),
                    settings=dict(bootstrap=dict(unit='family', reps=REPS, seed=SEED), no_refits=True, api_spend_usd=0, gpu_spend_usd=0,
                                  sources={k: str(v) for k, v in SRC.items()}))
    try:
        sets = load_sets(); results = {}; repro = {}
        for name, d in sets.items():
            print('SET', name, len(d['y']), int(d['y'].sum()), flush=True)
            results[name] = analyse(name, d)
            for det, r in results[name]['detectors'].items():
                repro[f'{name}/{det}'] = c = check(name, det, r)
                print('REPRO', name, det, round(r['scenario_round']['auc'], 4), c, flush=True)
        all_ok = all(c['auc'] and c['ci'] and c['pooled'] and c['paper_match'] for c in repro.values())
        write_json_atomic(run.dir/'robustness.json', dict(reproduction=repro, all_reproduced=all_ok, results=results))
        run.result(all_reproduced=all_ok); run.finish(); print(run.dir)
    except BaseException as e: run.finish('failed', str(e)); raise


if __name__ == '__main__': main()
