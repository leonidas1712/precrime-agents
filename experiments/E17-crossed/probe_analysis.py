"""E17 probe analysis, CPU only. Follows the README rules and replay_plan.json.

    uv run --no-project --with numpy --with scikit-learn python experiments/E17-crossed/probe_analysis.py \
        --calls private/crossed/e17_replay.calls.json --replay <downloaded replay run dir> --out private/crossed/e17_probe_R.json

Probe: the paper's last-prompt recipe (logistic regression, C=0.1, training-only StandardScaler), layer 48 (frozen in replay_plan.json).
- Test 1: PF forbidden (1) vs PF compromise (0). Leave-one-scenario-out: each scenario is scored by a probe trained on the other three.
  Within-scenario AUROC weighted by positives, episode bootstrap.
- Test 2: the same fold probes, unchanged, score PA allowed writes (never trained on) against PF forbidden out-of-fold scores.
- Added value: setup details alone vs setup details + probe score, both fit leave-one-scenario-out on the same rows. The probe score
  that the combiner trains on is itself out-of-fold (nested: inside each outer fold, the training rows get inner leave-one-scenario-out
  probe scores), so no row's probe score comes from a model that saw it. Paired bootstrap of the gain.
- Secondary: Test 1 at every saved layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_report import within_scenario_auroc  # noqa: E402

HEADLINE_LAYER = json.loads((HERE / 'replay_plan.json').read_text())['headline_layer']
N_BOOT, SEED = 1000, 20261105


def load(calls_path, replay_dir):
    calls = {c['call_id']: c for c in json.loads(Path(calls_path).read_text())}
    with open(Path(replay_dir) / 'calls.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(calls), (len(rows), len(calls))
    X, meta, layers = [], [], None
    for r in rows:
        z = np.load(Path(replay_dir) / r['activation_file'])
        layers = list(z['layer_indices']) if layers is None else layers
        X.append(z['residual_last'])
        meta.append(calls[r['call_id']])
    return np.stack(X), meta, layers   # X: (n_calls, n_layers, 5120)


def fit_probe(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X), y)
    return lambda Z: clf.decision_function(sc.transform(Z))


def loso_scores(X, y, scen, score_idx, train_idx):
    """For each scenario s: train on train_idx rows outside s, score score_idx rows inside s."""
    out = {}
    for s in sorted(set(scen[i] for i in score_idx)):
        tr = [i for i in train_idx if scen[i] != s]
        te = [i for i in score_idx if scen[i] == s]
        f = fit_probe(X[tr], y[tr])
        out.update(zip(te, f(X[te])))
    return out


def setup_matrix(meta, idx):
    keys = sorted({k for i in idx for k in meta[i]['setup']})
    return np.array([[meta[i]['setup'].get(k, 0.0) for k in keys] for i in idx])


def combo_scores(Xl, y, scen, meta, t1, with_probe):
    """Setup details (optionally + nested out-of-fold probe score), leave-one-scenario-out."""
    S = dict(zip(t1, setup_matrix(meta, t1)))
    out = {}
    for s in sorted(set(scen[i] for i in t1)):
        tr = [i for i in t1 if scen[i] != s]
        te = [i for i in t1 if scen[i] == s]
        A_tr, A_te = np.array([S[i] for i in tr]), np.array([S[i] for i in te])
        if with_probe:
            inner = loso_scores(Xl, y, scen, tr, tr)          # out-of-fold probe scores for the training rows
            outer = fit_probe(Xl[tr], y[tr])(Xl[te])          # test rows: probe trained on all training rows
            A_tr = np.column_stack([A_tr, [inner[i] for i in tr]])
            A_te = np.column_stack([A_te, outer])
        sc = StandardScaler().fit(A_tr)
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(A_tr), y[tr])
        out.update(zip(te, clf.decision_function(sc.transform(A_te))))
    return out


def wauc(idx, s, y, scen):
    return within_scenario_auroc([{'s': s[i], 'y': y[i], 'scen': scen[i]} for i in idx])


def boot_ci(idx, stat, rng):
    """Episode bootstrap (one row per episode), resampled within scenario so every scenario stays present."""
    by = {}
    for i in idx:
        by.setdefault(stat.scen[i], []).append(i)
    vals = []
    for _ in range(N_BOOT):
        b = [i for g in by.values() for i in rng.choice(g, len(g))]
        v = stat(b)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


class Stat:
    def __init__(self, fn, scen):
        self.fn, self.scen = fn, scen

    def __call__(self, idx):
        return self.fn(idx)


def main(a):
    X, meta, layers = load(a.calls, a.replay)
    n = len(meta)
    y = np.array([m['label_forbidden'] for m in meta])
    scen = [m['scenario'] for m in meta]
    rows = [m['row'] for m in meta]
    t1 = [i for i in range(n) if rows[i] in ('test1_forbidden', 'test1_compromise')]
    pa = [i for i in range(n) if rows[i] == 'test2_pa_allowed']
    pf_forb = [i for i in t1 if y[i] == 1]
    rng = np.random.default_rng(SEED)
    L = layers.index(HEADLINE_LAYER)
    Xl = X[:, L, :]
    R = {'n': {'test1_forbidden': len(pf_forb), 'test1_compromise': len(t1) - len(pf_forb), 'test2_pa_allowed': len(pa)},
         'headline_layer': HEADLINE_LAYER}

    # Test 1 + Test 2 scores from the same fold probes (trained on Test 1 rows only)
    s = loso_scores(Xl, y, scen, t1 + pa, t1)
    st1 = Stat(lambda idx: wauc(idx, s, y, scen), scen)
    R['test1'] = {'auroc': wauc(t1, s, y, scen), 'ci': boot_ci(t1, st1, rng),
                  'by_scenario': {sc: wauc([i for i in t1 if scen[i] == sc], s, y, scen) for sc in sorted(set(scen))}}
    t2 = pf_forb + pa   # PA allowed rows are label 0
    R['test2'] = {'auroc': wauc(t2, s, y, scen), 'ci': boot_ci(t2, Stat(lambda idx: wauc(idx, s, y, scen), scen), rng)}
    # Where PA allowed writes fall relative to PF compromises: probability that a PA allowed write scores higher
    # (more 'forbidden') than a PF compromise in the same scenario. 0.5 = the probe treats the two kinds of allowed write alike.
    pa_set = set(pa)
    is_pa = np.array([1 if i in pa_set else 0 for i in range(n)])
    R['test2']['pa_above_compromise'] = wauc([i for i in t1 if y[i] == 0] + pa, s, is_pa, scen)

    # Added value over the setup details
    s_setup = combo_scores(Xl, y, scen, meta, t1, with_probe=False)
    s_both = combo_scores(Xl, y, scen, meta, t1, with_probe=True)
    gain = Stat(lambda idx: (lambda a, b: None if a is None or b is None else a - b)(
        wauc(idx, s_both, y, scen), wauc(idx, s_setup, y, scen)), scen)
    R['added_value'] = {'setup_only': wauc(t1, s_setup, y, scen), 'setup_plus_probe': wauc(t1, s_both, y, scen),
                        'gain': gain(t1), 'gain_ci': boot_ci(t1, gain, rng),
                        'setup_only_ci': boot_ci(t1, Stat(lambda idx: wauc(idx, s_setup, y, scen), scen), rng)}

    # Secondary: layer sweep on Test 1 (no CIs)
    R['layer_sweep'] = {}
    for j, layer in enumerate(layers):
        sj = loso_scores(X[:, j, :], y, scen, t1, t1)
        R['layer_sweep'][int(layer)] = wauc(t1, sj, y, scen)

    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--calls', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--out', required=True)
    main(p.parse_args())
