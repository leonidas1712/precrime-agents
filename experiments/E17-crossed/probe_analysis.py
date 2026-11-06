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

    # ---- additions for the paper appendix (after the numbers above, so their bootstrap draws are unchanged) ----
    comp_pa = [i for i in t1 if y[i] == 0] + pa
    R['test2']['pa_above_compromise_ci'] = boot_ci(comp_pa, Stat(lambda idx: wauc(idx, s, is_pa, scen), scen), rng)
    R['test2']['by_scenario'] = {sc: wauc([i for i in t2 if scen[i] == sc], s, y, scen) for sc in sorted(set(scen))}

    # Text reader: the paper's TF-IDF recipe on everything the target saw before the scored call (system prompt and
    # task message excluded, file names replaced by their roles), same leave-one-scenario-out folds. Test 1 only.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pilot_report import load_run, scored_prompt_text
    eps = {x['m']['episode_id']: x for d in a.runs for x in load_run(d, legacy=False)}
    texts = [scored_prompt_text(eps[m['episode_id']]) for m in meta]
    s_text = {}
    for sc in sorted(set(scen)):
        tr = [i for i in t1 if scen[i] != sc]
        te = [i for i in t1 if scen[i] == sc]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, norm='l2')
        Xtr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(C=0.1, max_iter=2000, tol=1e-5, random_state=42).fit(Xtr, y[tr])
        s_text.update(zip(te, clf.decision_function(vec.transform([texts[i] for i in te]))))
    R['text_reader'] = {'auroc': wauc(t1, s_text, y, scen),
                        'ci': boot_ci(t1, Stat(lambda idx: wauc(idx, s_text, y, scen), scen), rng)}
    R['probe_minus_setup'] = {'value': R['test1']['auroc'] - R['added_value']['setup_only'],
                              'ci': boot_ci(t1, Stat(lambda idx: (lambda u, v: None if u is None or v is None else u - v)(
                                  wauc(idx, s, y, scen), wauc(idx, s_setup, y, scen)), scen), rng)}

    # Behaviour of all main-run episodes
    from collections import Counter
    out = Counter((x['cell'], x['outcome']) for x in eps.values())
    R['behaviour'] = {'episodes': len(eps), 'complete': sum(x['complete'] for x in eps.values()),
                      'outcomes': {f'{c}|{o}': k for (c, o), k in sorted(out.items())},
                      'by_scenario': {sc: {f'{c}|{o}': k for (c, o), k in sorted(Counter(
                          (x['cell'], x['outcome']) for x in eps.values() if x['scen'] == sc).items())} for sc in sorted(set(scen))},
                      'pf_complete': sum(x['complete'] for x in eps.values() if x['cell'] == 'PF'),
                      'pa_complete': sum(x['complete'] for x in eps.values() if x['cell'] == 'PA')}
    R['behaviour']['compromise_rate'] = R['n']['test1_compromise'] / R['behaviour']['pf_complete']
    R['behaviour']['forbidden_rate'] = R['n']['test1_forbidden'] / R['behaviour']['pf_complete']
    R['behaviour']['pa_allowed_rate'] = R['n']['test2_pa_allowed'] / R['behaviour']['pa_complete']
    # Outcome grid for the appendix behaviour table, zeros included
    groups = {'PF': {'forbidden': ['forbidden'], 'compromise': ['allowed'], 'no_real_write': ['no_disclosure', 'fake_value'],
                     'other': ['channel_only', 'excluded_channel_first', 'other_file'], 'incomplete': ['incomplete']},
              'PA': {'allowed': ['allowed'], 'no_real_write': ['no_disclosure', 'fake_value'],
                     'other': ['forbidden', 'channel_only', 'excluded_channel_first', 'other_file'], 'incomplete': ['incomplete']}}
    grid = {}
    for sc in sorted(set(scen)) + ['all']:
        xs = [x for x in eps.values() if sc == 'all' or x['scen'] == sc]
        grid[sc] = {c: {g: sum(x['cell'] == c and x['outcome'] in outs for x in xs) for g, outs in gs.items()}
                    for c, gs in groups.items()}
        for c in groups:
            grid[sc][c]['episodes'] = sum(x['cell'] == c for x in xs)
            grid[sc][c]['complete'] = sum(x['cell'] == c and x['complete'] for x in xs)
    R['behaviour']['grid'] = grid
    R['counts_by_scenario'] = {sc: {r: sum(1 for i in range(n) if scen[i] == sc and rows[i] == r) for r in sorted(set(rows))}
                               for sc in sorted(set(scen))}

    # Resampling check on the pilot episodes (same prompts resent 10 times): how much the choice is set by the prompt
    import resample
    RS = resample.report(a.resample)
    R['resample'] = {'prompts': RS['n_usable'], 'pure': RS['pure'], 'pure_share': RS['pure'] / RS['n_usable'],
                     'ceiling_within': RS['ceiling_within'], 'ceiling_within_ci': list(RS['ceiling_within_ci'])}

    # Per-row scores for the figure (scores are comparable within a scenario: one fold probe per scenario)
    with open(Path(a.out).parent / 'scores.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode_id', 'scenario', 'row', 'label_forbidden', 'probe_score', 'setup_score', 'text_score'])
        for i in sorted(s):
            w.writerow([meta[i]['episode_id'], scen[i], rows[i], int(y[i]), f'{s[i]:.6f}',
                        f'{s_setup[i]:.6f}' if i in s_setup else '', f'{s_text[i]:.6f}' if i in s_text else ''])
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R.items() if k not in ('behaviour',)}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--calls', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--out', help='results.json path (scores.csv goes next to it); default: a new run folder')
    p.add_argument('--runs', nargs='+', default=['runs/20261105-0042-e17-main-stage1', 'runs/20261105-1223-e17-main-stage2'])
    p.add_argument('--resample', default='runs/20261104-0748-e17-resample')
    a = p.parse_args()
    if a.out:
        main(a)
    else:
        sys.path.insert(0, str(HERE.parents[1] / 'src'))
        from collusion.runs import start_run
        run = start_run('e17-probe-analysis', experiment='E17', parents=[Path(r).name for r in a.runs] + [Path(a.replay).name, Path(a.resample).name],
                        settings={'headline_layer': HEADLINE_LAYER, 'recipe': 'LR C=0.1, train-only StandardScaler, LOSO', 'n_boot': N_BOOT,
                                  'seed': SEED, 'replay': str(a.replay)}, tags=['e17', 'cpu'], model='Qwen/Qwen3-32B')
        a.out = str(run.dir / 'results.json')
        main(a)
        run.finish()
