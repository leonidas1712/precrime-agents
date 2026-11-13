"""Task 4 (queue item 21): probe AUROC by layer for the headline per-call label, CPU only.

Validation by layer: saved in the frozen plans (paper-arxiv/frozen/*_percall_plan.json, models[].selection),
one inner 25%-family validation split per seed; this is what picked layers 48/16/48 (Qwen) and 6/6/13 (gpt-oss).
Test by layer (new): refit the frozen recipe (models.fit_logistic: train-only StandardScaler, C=0.1, round-balanced
weights) on all development training calls at each candidate layer, using the saved activations, and score the test
calls (Qwen 1,825 / gpt-oss 594) with the paper metric (within scenario x round AUROC, family bootstrap).
The final fit does not depend on the seed (lbfgs is deterministic), so test-by-layer is one number per layer;
the check below confirms that the refit at each frozen layer reproduces the frozen weights.

    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task4_layers.py (needs torch for loading saved activations)
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
from common_v2 import ROOT, FOLLOW, OUT, stream_metrics
sys.path.insert(0, str(ROOT / 'experiments/E05-probes'))
from models import fit_logistic, score  # noqa: E402

OLD = Path('/Users/leo/collusion-research')
REPS = 1000


def per_layer(name, x_train_fn, y, rounds, test_x_fn, test_rows, test_y, frozen, layers):
    res = {}
    for L in layers:
        m = fit_logistic(x_train_fn(L), y, rounds, 42)
        s = score(m, test_x_fn(L))
        met, _ = stream_metrics(test_rows, np.arange(len(test_rows)), test_y, s[None], np.zeros((1, len(test_rows)), int), REPS)
        entry = dict(test_auc=met['within_round_auc'], test_ci95=met['within_round_ci95'])
        for spec in frozen:
            if spec['layer'] == L:
                with gzip.open(ROOT / spec['weights'], 'rt') as f:
                    w = json.load(f)
                entry.setdefault('frozen_weight_check_max_abs_coef_diff', {})[spec['seed']] = float(np.max(np.abs(np.array(w['coef']) - m[1].coef_[0])))
        res[L] = entry
        print(name, L, json.dumps(entry, default=float), flush=True)
    return res


def qwen():
    plan = json.loads((ROOT / 'paper-arxiv/frozen/qwen_percall_plan.json').read_text())
    rows = json.loads((FOLLOW / plan['audit'] / 'rows.json').read_text())
    x = np.load(plan['feature_path'], mmap_mode='r')
    tr = np.array(plan['train_rows']); te = np.array(plan['test_rows'])
    ix = np.array([r['index'] for r in rows])
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows])
    test_rows = [rows[i] for i in te]
    out = per_layer('qwen', lambda L: x[ix[tr], L], y[tr], rounds[tr], lambda L: x[ix[te], L], test_rows, y[te], plan['models'], plan['recipe']['candidates'])
    return dict(train_calls=len(tr), train_positives=int(y[tr].sum()), test_calls=len(te), test_positives=int(y[te].sum()),
                test_families=len({r['family'] for r in test_rows}),
                chosen_layers=[m['layer'] for m in plan['models']],
                validation_by_seed={m['seed']: m['selection'] for m in plan['models']}, test_by_layer=out)


def gpt():
    plan = json.loads((ROOT / 'paper-arxiv/frozen/gpt_percall_plan.json').read_text())
    src = json.loads((OLD / 'runs/20261026-0522-e05-reasoning-stream-data/replies.json').read_text())
    rows = [r for r in src if r['stream_window'] == 'post_ask' and r['reasoning_tokens'] > 0]
    tr = np.array([i for i, r in enumerate(rows) if r['condition'] == 'recruiting']); assert len(tr) == 1995
    x = np.load(OLD / 'runs/20261025-2305-e05-gpt-oss-full-data/features.npy', mmap_mode='r')
    ix = np.array([r['index'] for r in rows]); y = np.array([r['y_immediate'] for r in rows]); rnd = np.array([r['round'] for r in rows])
    fresh = json.loads((ROOT / plan['fresh_rows']).read_text())
    for r in fresh:
        r['round'] = int(r['round']); r['y_immediate'] = int(r['y_immediate'])
    art = []
    for r in fresh:
        with np.load(r['artifact'], allow_pickle=False) as a:
            art.append(np.asarray(a['residual_last'], float))
    art = np.stack(art)  # (594, 24, 2880)
    ty = np.array([r['y_immediate'] for r in fresh])
    out = per_layer('gpt', lambda L: x[ix[tr], L], y[tr], rnd[tr], lambda L: art[:, L], fresh, ty, plan['models'], plan['recipe']['candidates'])
    return dict(train_calls=len(tr), train_positives=int(y[tr].sum()), test_calls=len(fresh), test_positives=int(ty.sum()),
                test_families=len({r['family'] for r in fresh}),
                chosen_layers=[m['layer'] for m in plan['models']],
                validation_by_seed={m['seed']: m['selection'] for m in plan['models']}, test_by_layer=out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    p = OUT / 'task4_layers.json'
    res = json.loads(p.read_text()) if p.exists() else {}
    if which in ('gpt', 'both'):
        res['gptoss'] = gpt(); p.write_text(json.dumps(res, indent=1, default=float))
    if which in ('qwen', 'both'):
        res['qwen'] = qwen(); p.write_text(json.dumps(res, indent=1, default=float))
    # reference: frozen ensemble test AUROC (paper)
    res['reference'] = dict(qwen_frozen_ensemble_test=json.loads((ROOT / 'runs/20261026-2212-e05-arxiv-qwen-fresh-percall/metrics.json').read_text())['within_round_auc'],
                            gptoss_frozen_ensemble_test=json.loads((ROOT / 'runs/20261026-2222-e05-arxiv-gpt-fresh-percall/metrics.json').read_text())['within_round_auc'])
    p.write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
