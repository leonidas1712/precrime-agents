"""Shared loaders for E18 robustness analyses (saved activations, CPU only).

Everything here wraps existing E05 code; nothing re-implements the probe recipe:
  - gpt-oss development fits: experiments/E05-probes/lastprompt_percall.py run_fit (matched 1,995-call cohort)
  - Qwen development fits:    experiments/E05-probes/arxiv_qwen_refit.py fit
  - fresh (test) protocol:    mirrors the freeze() loops of arxiv_frozen_percall.py / arxiv_gpt_fresh.py
  - metric:                   reasoning_stream_report.stream_metrics
The only change for the regularisation sweep is the C passed to sklearn's LogisticRegression, patched
into models.py / reasoning_stream_fit.py by set_C().
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E05 = REPO / 'experiments/E05-probes'
for p in [str(REPO / 'src'), str(E05), str(REPO / 'experiments/E01-behaviour-pilot')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression as _LR  # noqa: E402

OLD = Path('/Users/leo/collusion-research/runs')            # read-only: gpt-oss data, Qwen features
FUP = Path('/Users/leo/collusion-research-followup/runs')    # read-only: Qwen audit/refit, frozen test runs

# gpt-oss
GPT_DATA = OLD / '20261025-2305-e05-gpt-oss-full-data'
GPT_STREAM_DATA = OLD / '20261026-0522-e05-reasoning-stream-data'
GPT_FITS = OLD / '20261026-0138-e05-gpt-oss-main'
GPT_DEV_PLAN = OLD / '20261025-2251-e02-gpt-oss-full-plan'
GPT_TEST_PLAN = OLD / '20261026-1225-e02-gpt-oss-confirm-plan'
GPT_TEST_SCORES = OLD / '20261026-1302-e05-stream-confirm-score'
GPT_FREEZE = FUP / '20261026-2218-e05-arxiv-gpt-percall-freeze'
GPT_LAYERS = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 20, 22, 23]
# Qwen
Q_AUDIT = FUP / '20261026-2155-e05-qwen-any-public-audit'
Q_REFIT = FUP / '20261026-2201-e05-arxiv-qwen-any-public'
Q_FEATURES = '/Users/leo/collusion-research/runs/20261025-1026-e05-replay3-data-1d95/features.npy'
Q_PLAN = REPO / 'paper-arxiv/frozen/qwen_percall_plan.json'
GPT_PLAN = REPO / 'paper-arxiv/frozen/gpt_percall_plan.json'
Q_LAYERS = list(range(0, 64, 4)) + [63]
SEEDS = [42, 43, 44]

# Published values that Step 1 must reproduce (stored metrics files).
TARGETS = {
    'qwen_dev/eventual': (FUP / '20261026-2201-e05-arxiv-qwen-any-public/metrics.json', ['eventual/loso']),
    'qwen_dev/round_near': (FUP / '20261026-2201-e05-arxiv-qwen-any-public/metrics.json', ['round_near/loso']),
    'qwen_dev/immediate': (FUP / '20261026-2201-e05-arxiv-qwen-any-public/metrics.json', ['immediate/loso']),
    'qwen_dev/strict1': (FUP / '20261026-2223-e05-arxiv-qwen-strict-next/metrics.json', []),
    'gpt_dev/eventual_any': (FUP / '20261026-1304-e05-lastprompt-label-ladder/metrics.json', ['eventual_any/loso']),
    'gpt_dev/round_any': (FUP / '20261026-1304-e05-lastprompt-label-ladder/metrics.json', ['round_any/loso']),
    'gpt_dev/immediate': (FUP / '20261026-1221-e05-lastprompt-percall-6013/metrics.json', ['matched/immediate/loso']),
    'gpt_dev/strict1': (FUP / '20261026-1221-e05-lastprompt-percall-6013/metrics.json', ['matched/strict1/loso']),
    'qwen_test/immediate': (FUP / '20261026-2212-e05-arxiv-qwen-fresh-percall/metrics.json', []),
    'gpt_test/immediate': (FUP / '20261026-2222-e05-arxiv-gpt-fresh-percall/metrics.json', []),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target(key):
    path, keys = TARGETS[key]
    m = json.loads(path.read_text())
    for k in keys:
        m = m[k]
    return m


def set_C(C, max_iter=2000):
    """Patch the logistic C (and optionally max_iter) used by models.fit_logistic and reasoning_stream_fit.fit."""
    import models
    def factory(**kw):
        kw['C'] = C
        kw['max_iter'] = max_iter
        return _LR(**kw)
    models.LogisticRegression = factory
    try:
        import reasoning_stream_fit
        reasoning_stream_fit.LogisticRegression = factory
        reasoning_stream_fit.DATA = GPT_DATA
    except ImportError:
        pass


def summary(m):
    keep = ['within_round_auc', 'within_round_ci95', 'within_round_by_seed', 'pooled_auc', 'pooled_ci95',
            'positive_coverage_by_seed', 'bootstrap_groups', 'rows', 'positives', 'episodes',
            'within_round_valid_bootstrap_reps']
    return {k: m[k] for k in keep if k in m}


def strata_counts(rows, ids, y):
    """Scenario x round strata: total, and those with both classes (the ones the metric keeps)."""
    cells = {}
    for i, label in zip(ids, y):
        cells.setdefault((rows[i]['scenario'], rows[i]['round']), set()).add(int(label))
    mixed = [k for k, v in cells.items() if len(v) == 2]
    pos_in_mixed = sum(int(l) for i, l in zip(ids, y) if (rows[i]['scenario'], rows[i]['round']) in set(mixed))
    return dict(strata=len(cells), strata_with_both_classes=len(mixed), positives_in_kept_strata=int(pos_in_mixed))


# ---------------------------------------------------------------- gpt-oss development rows

def gpt_dev():
    """rows, base, x, plans exactly as lastprompt_percall.load(), with the ladder labels of lastprompt_label_ladder.py."""
    import lastprompt_percall as lp
    lp.DATA, lp.STREAM_DATA, lp.FITS = GPT_DATA, GPT_STREAM_DATA, GPT_FITS
    rows, base, x, plans = lp.load()
    for r in rows:  # lastprompt_label_ladder.main
        r['y_round_any'] = int(bool(r['ever_public']) and bool(r['y_near']))
        r['y_eventual_any'] = int(bool(r['ever_public']))
    return rows, base, x, plans


# ---------------------------------------------------------------- gpt-oss test rows

def gpt_test_rows():
    """Build Ever / round / strict-next labels for the 594 gpt-oss test calls the way the dev labels were built.

    Dev (reasoning_stream_data.prepare + reasoning_stream_labels.add_horizon_labels):
      ever_public  = plan episode first_public_emission_ordinal is not None
      y_near       = ever_public and 0 <= plan episode action_round - call round <= 1
      y_round_any  = ever_public and y_near
      eligible_replies_until_public = distance in eligible post-ask replies (empty reasoning counted) to the
                     leaking reply; y_strict1 = (distance == 1); own-leak calls dropped for strict1.
    Test: the same fields from the test plan (20261026-1225-e02-gpt-oss-confirm-plan/episodes.json, built
    by the same plan builder from the test episode files), over the post-ask selection that the frozen
    scorer used (20261026-1302-e05-stream-confirm-score/probe_scores.csv: eligible or leaking, post-ask).
    """
    from reasoning_stream_labels import add_horizon_labels
    eps = {e['episode']: e for e in json.loads((GPT_TEST_PLAN / 'episodes.json').read_text())}
    plan_rows = {r['row_id']: r for r in map(json.loads, (GPT_TEST_PLAN / 'rows.jsonl').read_text().splitlines())}
    sel = [r for r in csv.DictReader((GPT_TEST_SCORES / 'probe_scores.csv').open()) if r['condition'] == 'recruiting']
    assert len(sel) == 671
    rows = []
    for r in sel:
        e = eps[r['episode']]; pr = plan_rows[r['row_id']]
        ever = e['first_public_emission_ordinal'] is not None
        assert ever == (r['ever_public'] == 'True') == (pr['first_action'] is not None)
        if ever:
            assert pr['first_action'] == e['first_public_emission_ordinal']
            assert e['action_round'] == e['source_metadata']['first_public_emission_round'] == pr['action_round']
        assert pr['post_ask'] and int(r['round']) == pr['round'] and int(r['ordinal']) == pr['ordinal']
        # episode file is the one the plan hashed
        assert sha(OLD.parent / e['source_metadata']['episode_file']) == e['source_metadata']['episode_sha256']
        rnd = int(r['round'])
        rows.append(dict(row_id=r['row_id'], request_sha256=r['request_sha256'], episode=r['episode'], family=r['family'],
                         scenario=r['scenario'], condition='recruiting', round=rnd, ordinal=int(r['ordinal']),
                         reasoning_tokens=int(r['reasoning_tokens']), stream_window='post_ask', label=e['label'],
                         y_immediate=int(r['y_immediate']), ever_public=int(ever),
                         y_near=int(ever and 0 <= e['action_round'] - rnd <= 1)))
    add_horizon_labels(rows)
    for r in rows:
        r['y_round_any'] = int(bool(r['ever_public']) and bool(r['y_near']))
        r['y_eventual_any'] = int(bool(r['ever_public']))
        r['y_strict1'] = int(r['eligible_replies_until_public'] == 1)
    rows = [r for r in rows if r['reasoning_tokens'] > 0]
    fresh = json.loads((GPT_FREEZE / 'fresh_rows.json').read_text())
    assert [f['row_id'] for f in fresh] == [r['row_id'] for r in rows]  # same 594 calls, same order
    for f, r in zip(fresh, rows):
        assert int(f['y_immediate']) == r['y_immediate'] and f['request_sha256'] == r['request_sha256']
        r['artifact'], r['artifact_sha256'] = f['artifact'], f['artifact_sha256']
    return rows


def gpt_test_features(rows):
    x = np.zeros((len(rows), 24, 2880), dtype=np.float32)
    for j, r in enumerate(rows):
        assert sha(r['artifact']) == r['artifact_sha256']
        with np.load(r['artifact'], allow_pickle=False) as a:
            v = a['residual_last']; assert v.shape == (24, 2880) and np.isfinite(v).all(); x[j] = v
    return x


# ---------------------------------------------------------------- Qwen rows

def qwen_rows():
    """Audit rows (dev index < 6026, test index >= 6026), labels re-derived from the episode files."""
    from request_labels import request_index_labels, completed_target_requests
    assert sha(Q_AUDIT / 'rows.json') == json.loads(Q_PLAN.read_text())['audit_rows_sha256']
    rows = json.loads((Q_AUDIT / 'rows.json').read_text())
    audit = {e['episode']: e for e in json.loads((Q_AUDIT / 'episode_audit.json').read_text())}
    cache = {}
    for r in rows:  # same rule as qwen_any_public.audit, recomputed from the episode JSON
        if r['episode'] not in cache:
            folder, name = r['episode'].split('/')
            path = FUP / folder / 'episodes' / name
            assert sha(path) == audit[r['episode']]['episode_sha256']
            ep = json.loads(path.read_text()); b = request_index_labels(ep, ep.get('label'))
            action = b['first_action_target_request']
            action_round = next((rec['round'] for o, (_, rec) in enumerate(completed_target_requests(ep), 1) if o == action), None)
            cache[r['episode']] = (action, action_round)
        action, action_round = cache[r['episode']]
        assert r['public_ordinal'] == action and r['public_round'] == action_round
        assert r['y_immediate'] == int(r['ordinal'] == action)
        assert r['y_eventual'] == int(action is not None)
        assert r['y_round_near'] == int(action is not None and 0 <= action_round - r['round'] <= 1)
        # strict next (arxiv_qwen_strict_next.py)
        r['y_strict1'] = int(r['public_ordinal'] is not None and r['public_ordinal'] - r['ordinal'] == 1)
        # channel-outcome round label (paper's original A_hazard: near_term.label_rows), from replay3 row fields
        if r['label']:
            assert r['action_round'] - r['round'] >= 0
        r['y_round_channel'] = int(bool(r['label']) and 0 <= r['action_round'] - r['round'] <= 1)
    return rows


# ---------------------------------------------------------------- fresh protocol (dev-only fit, test once)

def fresh_fit(rows, xget, train, cal, y, layers, seeds=SEEDS):
    """The freeze() loop of arxiv_frozen_percall.py / arxiv_gpt_fresh.py, parameterised by label vector y.

    xget(indices, layer) -> features. Returns one spec per seed: layer, inner choices, weights, training mean/SD.
    """
    from models import fit_logistic, score
    from metrics import within_round
    from replay2_core import split_inner
    rnd = np.array([r['round'] for r in rows]); sc = np.array([r['scenario'] for r in rows])
    specs = []
    for seed in seeds:
        f, v, _ = split_inner(rows, train, seed, False); choices = {}
        for layer in layers:
            m = fit_logistic(xget(f, layer), y[f], rnd[f], seed)
            choices[layer] = within_round(y[v], score(m, xget(v, layer)), rnd[v], stratum_key=sc[v])[0]
        assert any(np.isfinite(c) for c in choices.values())
        layer = max(layers, key=lambda k: (choices[k] if np.isfinite(choices[k]) else -1, -k))
        m = fit_logistic(xget(train, layer), y[train], rnd[train], seed); tr = score(m, xget(train, layer))
        scaler, lr = m
        specs.append(dict(seed=seed, layer=layer, selection=choices, train_mean=float(tr.mean()), train_sd=float(tr.std()),
                          mean=scaler.mean_, scale=scaler.scale_, coef=lr.coef_[0], intercept=float(lr.intercept_[0]),
                          inner_fit=len(f), inner_val=len(v), inner_val_positives=int(y[v].sum())))
    return specs


def fresh_score(specs, xtest_by_layer):
    """Mean of the seed logits standardised by each model's training mean/SD (frozen plans' aggregation)."""
    vals = []
    for s in specs:
        z = ((np.asarray(xtest_by_layer(s['layer']), float) - s['mean']) / s['scale']) @ s['coef'] + s['intercept']
        vals.append((z - s['train_mean']) / s['train_sd'])
    return np.mean(vals, axis=0)


def public_spec(s):
    return {k: v for k, v in s.items() if k not in ('mean', 'scale', 'coef')}


def qwen_fresh(rows, label, drop_immediate=False, reps=1000):
    """Fit on Qwen development recruiting calls (index < 6026), score the 1,825 test calls once."""
    from reasoning_stream_report import stream_metrics
    x = np.load(Q_FEATURES, mmap_mode='r'); idx = np.array([r['index'] for r in rows]); y = np.array([r['y_' + label] for r in rows])
    keep = lambda r: r['condition'] == 'recruiting' and not (drop_immediate and r['y_immediate'])
    train = np.array([i for i, r in enumerate(rows) if r['index'] < 6026 and keep(r)])
    test = np.array([i for i, r in enumerate(rows) if r['index'] >= 6026 and keep(r)])
    assert not {rows[i]['family'] for i in train} & {rows[i]['family'] for i in test}
    assert len(train) == (2910 if drop_immediate else 2976) and len(test) == (1787 if drop_immediate else 1825)
    specs = fresh_fit(rows, lambda ids, layer: x[idx[ids], layer], train, None, y, Q_LAYERS)
    values = fresh_score(specs, lambda layer: x[idx[test], layer])
    m, _ = stream_metrics(rows, test, y[test], values[None, :], np.zeros((1, len(test)), int), reps)
    m.update(strata_counts(rows, test, y[test]), train_calls=len(train), train_positives=int(y[train].sum()))
    return m, specs, test, values


def gpt_dev_fresh_rows():
    """The 1,995 + 892 rows that arxiv_gpt_fresh.freeze trains/calibrates on, with ladder labels."""
    from reasoning_stream_labels import add_horizon_labels
    rows = add_horizon_labels(json.loads((GPT_STREAM_DATA / 'replies.json').read_text()))
    rows = [r for r in rows if r['stream_window'] == 'post_ask' and r['reasoning_tokens'] > 0]
    for r in rows:
        r['y_round_any'] = int(bool(r['ever_public']) and bool(r['y_near']))
        r['y_eventual_any'] = int(bool(r['ever_public']))
        r['y_strict1'] = int(r['eligible_replies_until_public'] == 1)
    return rows


def gpt_fresh(dev, test_rows, xtest, label, drop_immediate=False, reps=1000):
    """Fit on gpt-oss development calls (matched cohort), score the 594 test calls once."""
    from reasoning_stream_report import stream_metrics
    x = np.load(GPT_DATA / 'features.npy', mmap_mode='r'); ix = np.array([r['index'] for r in dev])
    y = np.array([r['y_' + label] for r in dev])
    train = np.array([i for i, r in enumerate(dev) if r['condition'] == 'recruiting' and not (drop_immediate and r['y_immediate'])])
    assert len(train) == (1874 if drop_immediate else 1995)
    tk = np.array([j for j, r in enumerate(test_rows) if not (drop_immediate and r['y_immediate'])])
    assert len(tk) == (568 if drop_immediate else 594)
    assert not {str(dev[i]['family']) for i in train} & {str(test_rows[j]['family']) for j in tk}
    specs = fresh_fit(dev, lambda ids, layer: x[ix[ids], layer], train, None, y, GPT_LAYERS)
    values = fresh_score(specs, lambda layer: xtest[tk, layer])
    sub = [test_rows[j] for j in tk]; yt = np.array([r['y_' + label] for r in sub]); ids = np.arange(len(sub))
    m, _ = stream_metrics(sub, ids, yt, values[None, :], np.zeros((1, len(sub)), int), reps)
    m.update(strata_counts(sub, ids, yt), train_calls=len(train), train_positives=int(y[train].sum()))
    return m, specs, tk, values
