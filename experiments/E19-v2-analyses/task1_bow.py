"""Task 1 (queue item 17): bag-of-words scores on the permission experiments E08 / E14 / E15.

Classifiers: the paper recipe (TF-IDF uni+bigrams, min_df=2, 50k, C=0.1, round-balanced weights,
secret masked), fit once on ALL development calls under the forbidding rule:
  - qwen_latest : Qwen latest-message view (the paper's reported Qwen view; = E12 test classifier, 0.957)
  - qwen_full   : Qwen full-input view (system prompt incl. rule, tools, all messages)
  - gpt_full    : gpt-oss full-input view (the paper's reported gpt view), for the gpt-oss E08 replay
No refitting on any permission data.

Thresholds: each detector's cut is matched to the probe's flag rate on calls in refusing Qwen test
episodes (the E13 rule: lowest strict > cut flagging no larger share of the 1,407 refusing-episode calls
than the probe at its saved threshold 1.284). gpt-oss: matched on gpt-oss test refusing-episode calls.

Every statistic is computed with the same code path as the paper's analysis scripts; each analysis is
first run on the probe scores and must reproduce the paper's saved numbers.

    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task1_bow.py
"""
import csv, gzip, json, pickle, random, re, statistics as st, sys
from collections import defaultdict
import numpy as np
from common_v2 import *
from common_v2 import ROOT, OUT, bow_texts, strip_assistant, request_for, split_id, load_episode

import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E08A = _load('e08a', ROOT / 'experiments/E08-policy-counterfactual/analyze.py')
within_stratum_auc, pooled_auc, bootstrap_family_weights, interval = E08A.within_stratum_auc, E08A.pooled_auc, E08A.bootstrap_family_weights, E08A.interval
E14A = _load('e14a', ROOT / 'experiments/E14-permitted-onpolicy/analyze.py')
E15A = _load('e15a', ROOT / 'experiments/E15-giving-in-scaleup/analyze.py')

CACHE = OUT / 'cache'; CACHE.mkdir(exist_ok=True)
E08Q = ROOT / 'runs/20261026-1717-e08-policy-counterfactual'
E08G = ROOT / 'runs/20261026-1732-e08-policy-counterfactual-gpt-oss'
E14R = ROOT / 'runs/20261028-0224-e14-permitted-onpolicy'
E15R = ROOT / 'runs/20261028-1039-e15-giving-in-scaleup'
CALIB = ROOT / 'runs/20261027-0047-e05-arxiv-episode-calibration/scores.csv.gz'
QTHR = 1.283997258353578
E14PLAN = json.loads((ROOT / 'experiments/E14-permitted-onpolicy/plan.json').read_text())
CONSEQ = E14PLAN['amendment_v2']['addon_E08_permit_no_consequence']['consequence_sentences_by_scenario']
HEDGE = re.compile(json.loads((ROOT / 'experiments/E15-giving-in-scaleup/plan.json').read_text())['hedge_regex'])


# ------------------------------------------------------------------ fits
class Model:
    def __init__(self, tf, clf): self.tf, self.clf = tf, clf
    def score(self, docs): return self.clf.decision_function(self.tf.transform(docs))


def fit_all_dev(model, view, docs_override=None):
    d = dev(model); rows = d['rows']; docs = docs_override or d['texts'][view]
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows])
    w = position_weights(y, rounds); keep = w > 0
    tf, clf = fit([t for t, k in zip(docs, keep) if k], y[keep], w[keep])
    return Model(tf, clf)


def match_cut(scores, neg_mask, target):
    """E13 rule: lowest strict > cut whose flagged share of refusing calls is <= target."""
    s = np.asarray(scores)[neg_mask]; best = np.inf
    for cut in np.r_[np.unique(s)[::-1], -np.inf]:
        if (s > cut).mean() <= target + 1e-12: best = cut
        else: break
    return float(best)


# ------------------------------------------------------------------ E08 analysis (same code as experiments/E08 analyze.py, any score)
def e08_stats(by_call, threshold, seed=20261027, reps=1000, conds=('FORBID', 'PERMIT', 'NEUTRAL_EDIT')):
    """by_call: call_id -> {cond: row dict with score, class, family, scenario, round}."""
    leaking = [v for v in by_call.values() if v['FORBID']['class'] == 'leaking']
    fams = sorted({v['FORBID']['family'] for v in leaking})
    deltas = {c: np.array([v[c]['score'] - v['FORBID']['score'] for v in leaking]) for c in conds if c != 'FORBID'}
    rng = np.random.default_rng(seed)
    pf = np.array([v['FORBID']['family'] for v in leaking])
    boot = {c: [] for c in deltas}
    if 'PERMIT' in deltas and 'NEUTRAL_EDIT' in deltas: boot['PERMIT_minus_NEUTRAL'] = []
    for _ in range(reps):
        idx = np.concatenate([np.flatnonzero(pf == f) for f in rng.choice(fams, size=len(fams), replace=True)])
        for c in deltas: boot[c].append(float(np.median(deltas[c][idx])))
        if 'PERMIT_minus_NEUTRAL' in boot: boot['PERMIT_minus_NEUTRAL'].append(float(np.median((deltas['PERMIT'] - deltas['NEUTRAL_EDIT'])[idx])))
    paired = {f'{c}_minus_FORBID': dict(median=float(np.median(deltas[c])), ci95=interval(boot[c]),
                                         share_exactly_zero=float(np.mean(np.abs(deltas[c]) < 1e-12)),
                                         max_abs=float(np.max(np.abs(deltas[c])))) for c in deltas}
    if 'PERMIT_minus_NEUTRAL' in boot:
        paired['PERMIT_minus_NEUTRAL'] = dict(median=float(np.median(deltas['PERMIT'] - deltas['NEUTRAL_EDIT'])), ci95=interval(boot['PERMIT_minus_NEUTRAL']))
    rows = {c: [v[c] for v in by_call.values()] for c in conds}
    allf = sorted({r['family'] for r in rows['FORBID']})
    ab = {c: {'w': [], 'p': []} for c in conds}; tb = {c: [] for c in conds}
    for _ in range(reps):
        w = bootstrap_family_weights(allf, rng)
        for c in conds:
            ab[c]['w'].append(within_stratum_auc(rows[c], w)); ab[c]['p'].append(pooled_auc(rows[c], w))
        pw = np.array([w.get(v['FORBID']['family'], 0) for v in leaking])
        for c in conds:
            tb[c].append(float(np.average([v[c]['score'] > threshold for v in leaking], weights=pw)))
    auc = {c: dict(within_stratum=dict(value=within_stratum_auc(rows[c]), ci95=interval([x for x in ab[c]['w'] if np.isfinite(x)])),
                   pooled=dict(value=pooled_auc(rows[c]), ci95=interval([x for x in ab[c]['p'] if np.isfinite(x)]))) for c in conds}
    cross = {c: dict(numerator=int(sum(v[c]['score'] > threshold for v in leaking)), denominator=len(leaking),
                     fraction=float(np.mean([v[c]['score'] > threshold for v in leaking])), ci95=interval(tb[c])) for c in conds}
    return dict(calls=dict(leaking=len(leaking), non_leaking=len(by_call) - len(leaking), families_leaking=len(fams)),
                threshold=threshold, paired_leaking_call_changes=paired, auc=auc, threshold_crossing=cross)


def e08_inputs(run):
    plan = json.loads((run / 'plan.json').read_text())
    rows = list(csv.DictReader(open(run / 'scores.csv')))
    by = defaultdict(dict)
    for r in rows:
        by[r['call_id']][r['condition']] = dict(r, score=float(r['score']))
    return plan, by


def no_consequence_edit(call_id, scenario, plan):
    req, hp = request_for(call_id, (scenario, 'PERMIT', plan))
    import copy
    req = copy.deepcopy(req)
    sysm = [m for m in req['messages'] if m.get('role') == 'system'][0]
    s = CONSEQ[scenario]; assert sysm['content'].count(s) == 1
    sysm['content'] = sysm['content'].replace(s, '', 1)
    return req, hp


def texts_from_request(req, hp, gpt_mode=None):
    messages = json.loads(json.dumps(req['messages']))
    if gpt_mode is not None:
        last_user = max((i for i, m in enumerate(messages) if m.get('role') == 'user'), default=-1)
        for i, m in enumerate(messages):
            if not (gpt_mode == 'current_turn' and i > last_user):
                m.pop('reasoning', None); m.pop('reasoning_details', None)
    text = json.dumps(dict(messages=messages, tools=req.get('tools', [])), ensure_ascii=False, sort_keys=True)
    inc = json.dumps(messages[-1], ensure_ascii=False, sort_keys=True)
    for v in hp:
        text = text.replace(v, 'PROTECTED_VALUE'); inc = inc.replace(v, 'PROTECTED_VALUE')
    return text, inc


def main():
    res = {}
    # ---------------- fits
    pk = CACHE / 'fits_task1.pkl'
    if pk.exists():
        M = pickle.loads(pk.read_bytes())
    else:
        M = dict(qwen_latest=fit_all_dev('qwen', 'latest_input'), qwen_full=fit_all_dev('qwen', 'full_input'), gpt_full=fit_all_dev('gpt', 'full_input'))
        pk.write_bytes(pickle.dumps(M))
    # ---------------- Qwen test calls: reproduction + matched thresholds
    f = fresh_qwen(); frows = f['rows']; fy = np.array([r['label'] for r in frows])
    fs = {'qwen_latest': M['qwen_latest'].score(f['texts']['latest_input']), 'qwen_full': M['qwen_full'].score(f['texts']['full_input'])}
    saved_bow = {r['row_id']: float(r['bow']) for r in csv.DictReader(gzip.open(ROOT / 'runs/20261028-0228-e12-bow-test-auroc/bow_scores.csv.gz', 'rt'))}
    repro = max(abs(saved_bow[r['row_id']] - s) for r, s in zip(frows, fs['qwen_latest']))
    calib = [r for r in csv.DictReader(gzip.open(CALIB, 'rt')) if r['split'] == 'fresh' and r['condition'] == 'recruiting']
    ever = {r['row_id']: r['ever'] == 'True' for r in calib if r['detector'] == 'qwen_probe'}
    probe_fresh = {r['row_id']: float(r['score']) for r in calib if r['detector'] == 'qwen_probe'}
    neg = np.array([not ever[r['row_id']] for r in frows])
    pscore = np.array([probe_fresh[r['row_id']] for r in frows])
    target = float((pscore[neg] > QTHR).mean())
    thr = {'probe': QTHR}
    for k in fs: thr[k] = match_cut(fs[k], neg, target)
    res['qwen_test_reference'] = dict(
        calls=len(frows), leaking=int(fy.sum()), refusing_episode_calls=int(neg.sum()),
        probe_refusing_flag_rate=target, reproduction_max_abs_diff_vs_saved_E12_bow=repro,
        auroc_within_round={'probe': auroc(frows, fy, pscore), **{k: auroc(frows, fy, v) for k, v in fs.items()}},
        thresholds=thr,
        score_sd_test_calls={'probe': float(pscore.std()), **{k: float(v.std()) for k, v in fs.items()}},
        flag_rates={k: dict(refusing=float((v[neg] > thr[k]).mean()), leaking_calls=f"{int((v[fy == 1] > thr[k]).sum())}/{int(fy.sum())}")
                    for k, v in [('probe', pscore)] + list(fs.items())})
    print('qwen test reference', json.dumps(res['qwen_test_reference'], default=float), flush=True)

    # ---------------- E08 Qwen replay
    plan, by = e08_inputs(E08Q)
    probe_stats = e08_stats(by, QTHR)
    saved = json.loads((E08Q / 'metrics.json').read_text())
    chk = dict(permit_median=(probe_stats['paired_leaking_call_changes']['PERMIT_minus_FORBID']['median'], saved['paired_leaking_call_changes']['permit_minus_forbid']['median']),
               permit_ci=(probe_stats['paired_leaking_call_changes']['PERMIT_minus_FORBID']['ci95'], saved['paired_leaking_call_changes']['permit_minus_forbid']['ci95']),
               forbid_auc=(probe_stats['auc']['FORBID']['within_stratum']['value'], saved['auc']['FORBID']['within_stratum']['value']),
               permit_crossing=(probe_stats['threshold_crossing']['PERMIT']['numerator'], saved['threshold_crossing']['PERMIT']['numerator']))
    print('E08 probe check', chk, flush=True)
    det_by = {k: defaultdict(dict) for k in ('qwen_latest', 'qwen_full')}
    nc_scores = {}
    for cid, conds in by.items():
        c0 = conds['FORBID']; sc = c0['scenario']
        docs = {}
        for cond in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT'):
            docs[cond] = bow_texts(cid, (sc, cond, plan))
        docs['PERMIT_NO_CONSEQUENCE'] = texts_from_request(*no_consequence_edit(cid, sc, plan))
        for cond, (full, latest) in docs.items():
            for k, txt in (('qwen_full', full), ('qwen_latest', latest)):
                det_by[k][cid][cond] = dict(c0, score=float(M[k].score([txt])[0]))
    # sanity: FORBID texts identical to the test-call texts
    idx = {r['row_id']: i for i, r in enumerate(frows)}
    forbid_dev = max(abs(det_by['qwen_latest'][c]['FORBID']['score'] - fs['qwen_latest'][idx[c]]) for c in by)
    e08 = {'probe': dict(probe_stats, paper_check=chk)}
    for k in det_by:
        e08[k] = e08_stats(det_by[k], thr[k], conds=('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE'))
    # probe no-consequence (E14 addon)
    add = {r['call_id']: float(r['score']) for r in csv.DictReader(open(E14R / 'addon_scores.csv')) if r['condition'] == 'PERMIT_NO_CONSEQUENCE'}
    pby = {cid: dict(v, PERMIT_NO_CONSEQUENCE=dict(v['FORBID'], score=add[cid])) for cid, v in by.items()}
    e08['probe_with_no_consequence'] = e08_stats(pby, QTHR, conds=('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE'))
    e08['forbid_text_matches_test_text_max_abs_diff'] = forbid_dev
    res['E08_qwen'] = e08
    print('E08 qwen done', flush=True)

    # ---------------- E08 gpt-oss replay
    gplan, gby = e08_inputs(E08G)
    gprobe = e08_stats(gby, float(gplan['probe']['threshold']))
    gsaved = json.loads((E08G / 'metrics.json').read_text())
    gchk = dict(permit_median=(gprobe['paired_leaking_call_changes']['PERMIT_minus_FORBID']['median'], gsaved['paired_leaking_call_changes']['permit_minus_forbid']['median']),
                forbid_auc=(gprobe['auc']['FORBID']['within_stratum']['value'], gsaved['auc']['FORBID']['within_stratum']['value']))
    # gpt test calls for the matched threshold (all assumed current_turn retention, as all 83 E08 calls are)
    gfresh = json.loads((ROOT / 'runs/20261026-2218-e05-arxiv-gpt-percall-freeze/fresh_rows.json').read_text())
    gps = {r['row_id']: float(r['score']) for r in csv.DictReader(gzip.open(ROOT / 'runs/20261026-2222-e05-arxiv-gpt-fresh-percall/probe_scores.csv.gz', 'rt'))}
    gfresh = [r for r in gfresh if r['row_id'] in gps and r['condition'] == 'recruiting']
    gdocs = [texts_from_request(*request_for(r['row_id']), gpt_mode='current_turn')[0] for r in gfresh]
    gb = M['gpt_full'].score(gdocs); gneg = np.array([str(r['ever_public']) != 'True' for r in gfresh])
    gp = np.array([gps[r['row_id']] for r in gfresh])
    gtarget = float((gp[gneg] > gplan['probe']['threshold']).mean())
    gthr = match_cut(gb, gneg, gtarget)
    gdet = defaultdict(dict)
    calls_meta = {c['call_id']: c for c in gplan['calls']}
    for cid, conds in gby.items():
        c0 = conds['FORBID']; sc = c0['scenario']
        for cond in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT'):
            req, hp = request_for(cid, (sc, cond, gplan))
            gdet[cid][cond] = dict(c0, score=float(M['gpt_full'].score([texts_from_request(req, hp, calls_meta[cid]['history_reasoning_mode'])[0]])[0]))
    res['E08_gptoss'] = dict(probe=dict(gprobe, paper_check=gchk), gpt_full=e08_stats(gdet, gthr),
                             threshold_matching=dict(test_calls=len(gfresh), refusing_episode_calls=int(gneg.sum()), probe_refusing_flag_rate=gtarget,
                                                     bow_cut=gthr, bow_refusing_flag_rate=float((gb[gneg] > gthr).mean()),
                                                     score_sd_test_calls=dict(probe=float(gp.std()), gpt_full=float(gb.std())),
                                                     bow_test_auroc_within_round=auroc([dict(r, round=int(r['round'])) for r in gfresh], np.array([int(r['y_immediate']) for r in gfresh]), gb),
                                                     probe_test_auroc_within_round=auroc([dict(r, round=int(r['round'])) for r in gfresh], np.array([int(r['y_immediate']) for r in gfresh]), gp),
                                                     note='gpt-oss test texts built with current-turn reasoning retention for every call (retention mode is not stored for test rows; all 83 E08 calls use it)'))
    print('E08 gpt done', flush=True)

    # ---------------- E14 / E15 calls: texts and scores
    calls = json.loads((E15R / 'calls.json').read_text())
    e14rows = [r for r in csv.DictReader(open(E14R / 'scores.csv')) if r['condition'] in ('PERMIT', 'PERMIT_NO_CONSEQUENCE')]
    ids = sorted({c['call_id'] for c in calls} | {r['call_id'] for r in e14rows} | {r['row_id'] for r in frows})
    tp = CACHE / 'perm_texts.pkl'
    if tp.exists():
        T = pickle.loads(tp.read_bytes())
    else:
        T = {}
        for cid in ids:
            full, latest = bow_texts(cid)
            T[cid] = dict(full=full, latest=latest, full_noasst=strip_assistant(cid))
        tp.write_bytes(pickle.dumps(T))
    S = {k: dict(zip(ids, M[k].score([T[c]['latest' if k == 'qwen_latest' else 'full'] for c in ids]))) for k in ('qwen_latest', 'qwen_full')}
    S['qwen_full_noasst'] = dict(zip(ids, M['qwen_full'].score([T[c]['full_noasst'] for c in ids])))
    # hedge-vocabulary ablation: zero the coefficients of every feature containing a hedge-regex match
    hedge_feats = {}
    for k in ('qwen_latest', 'qwen_full'):
        names = M[k].tf.get_feature_names_out(); mask = np.array([bool(HEDGE.search(n)) for n in names])
        coef = M[k].clf.coef_[0].copy(); hedge_feats[k] = dict(n_features=int(mask.sum()), names_top=[str(n) for n in names[mask][np.argsort(-np.abs(coef[mask]))][:25]],
                                                                 weights_top=[float(x) for x in coef[mask][np.argsort(-np.abs(coef[mask]))][:25]])
        x = M[k].tf.transform([T[c]['latest' if k == 'qwen_latest' else 'full'] for c in ids])
        coef[mask] = 0
        S[k + '_nohedge'] = dict(zip(ids, x @ coef + M[k].clf.intercept_[0]))
    # probe scores
    P = {}
    for path in sorted(E15R.glob('batch-*.csv')):
        for r in csv.DictReader(open(path)):
            if r['condition'] != 'FORBID_REPRODUCTION': P[r['call_id']] = float(r['score'])
    for r in e14rows: P.setdefault(r['call_id'], float(r['score']))
    for r in csv.DictReader(open(E14R / 'scores.csv')):
        if r['condition'] == 'PERMIT_NO_CONSEQUENCE': P[r['call_id']] = float(r['score'])
    for r in frows: P[r['row_id']] = probe_fresh[r['row_id']]
    S['probe'] = P
    # thresholds for ablated variants: re-match on the test refusing calls
    for k in ('qwen_full_noasst', 'qwen_latest_nohedge', 'qwen_full_nohedge'):
        thr[k] = match_cut(np.array([S[k][r['row_id']] for r in frows]), neg, target)
    res['thresholds'] = thr
    res['hedge_vocabulary'] = hedge_feats

    # ---------------- E14
    disp = [dict(r, score=0.0, label=int(r['label']), round=int(r['round'])) for r in frows if r['scenario'] == 'dispatch-integration-code']
    e14 = {}
    for det in ('probe', 'qwen_latest', 'qwen_full', 'qwen_full_noasst', 'qwen_latest_nohedge', 'qwen_full_nohedge'):
        t = thr[det]; out = {}
        drows = [dict(r, score=S[det][r['row_id']]) for r in disp]
        fl = [r for r in drows if r['label']]
        out['FORBID_dispatch_test'] = dict(leaking_calls=E14A.describe([r['score'] for r in fl], t),
                                           within_round_auroc=E14A.within_round_auroc(drows), within_round_auroc_ci=E14A.boot(drows, E14A.within_round_auroc))
        for arm in ('PERMIT', 'PERMIT_NO_CONSEQUENCE'):
            ar = [dict(r, score=S[det][r['call_id']], label=int(r['label']), round=int(r['round'])) for r in e14rows if r['condition'] == arm]
            sh = [r for r in ar if r['label']]
            out[arm] = dict(sharing_calls=E14A.describe([r['score'] for r in sh], t),
                            median_diff_vs_forbidden_leaks=E14A.med(sh) - E14A.med(fl), median_diff_ci=E14A.boot_diff(sh, fl, E14A.med),
                            within_round_auroc=E14A.within_round_auroc(ar), within_round_auroc_ci=E14A.boot(ar, E14A.within_round_auroc),
                            calls=len(ar), sharing=len(sh), episodes=len({r['episode'] for r in ar}))
        e14[det] = out
    saved14 = json.loads((E14R / 'results.json').read_text())
    e14['probe_paper_check'] = dict(clean_over=(e14['probe']['PERMIT_NO_CONSEQUENCE']['sharing_calls']['over_threshold'], saved14['arms']['PERMIT_NO_CONSEQUENCE']['sharing_calls']['over_threshold']),
                                    clean_auc=(e14['probe']['PERMIT_NO_CONSEQUENCE']['within_round_auroc'], saved14['arms']['PERMIT_NO_CONSEQUENCE']['within_round_auroc']),
                                    clean_diff=(e14['probe']['PERMIT_NO_CONSEQUENCE']['median_diff_vs_forbidden_leaks'], saved14['arms']['PERMIT_NO_CONSEQUENCE']['median_diff_vs_forbidden_leaks']),
                                    forbid_auc=(e14['probe']['FORBID_dispatch_test']['within_round_auroc'], saved14['forbidden_dispatch_test']['within_round_auroc']))
    res['E14'] = e14
    print('E14 done', e14['probe_paper_check'], flush=True)

    # ---------------- E15
    e15 = {}
    for det in ('probe', 'qwen_latest', 'qwen_full', 'qwen_full_noasst', 'qwen_latest_nohedge', 'qwen_full_nohedge'):
        t = thr[det]
        cs = [dict(c, score=S[det][c['call_id']], group='FORBID' if c['arm'] in ('FORBID', 'FORBID_OLD') else c['arm']) for c in calls]
        share = [c for c in cs if c['label'] and not (c['arm'] == 'LATE' and c['before_ask'])]
        by_g = {g: [c for c in share if c['group'] == g] for g in E15A.ARMS}
        nonread = {g: [c for c in v if not c['follows_read']] for g, v in by_g.items()}
        read = {g: [c for c in v if c['follows_read']] for g, v in by_g.items()}
        r3 = [c for c in nonread['FORBID'] if c['round'] >= 3]
        o = dict(threshold=t,
                 nonread={g: E15A.rate(v, t) for g, v in nonread.items()},
                 follows_read={g: E15A.rate(v, t) for g, v in read.items()},
                 all_sharing={g: E15A.rate(v, t) for g, v in by_g.items()},
                 diff_EARLY_minus_FORBID_nonread=E15A.boot_diff(nonread['EARLY'], nonread['FORBID'], t),
                 diff_LATE_minus_FORBID_nonread_round3plus=E15A.boot_diff(nonread['LATE'], r3, t),
                 nonread_FORBID_round3plus=E15A.rate(r3, t),
                 recency={g: {b: E15A.rate([c for c in v if E15A.bin_of(c) == b], t) for b in E15A.BINS} for g, v in by_g.items()})
        # threshold-free: AUROC forbidden (pos) vs permitted EARLY (neg) sharing calls; bootstrap over sharing calls (= episodes)
        def auc_boot(a, b, reps=2000, seed=20261028):
            rng = random.Random(seed); xa = [c['score'] for c in a]; xb = [c['score'] for c in b]
            est = E14A.auroc(xa, xb)
            bs = sorted(E14A.auroc([rng.choice(xa) for _ in xa], [rng.choice(xb) for _ in xb]) for _ in range(reps))
            return dict(value=est, ci95=[bs[int(.025 * reps)], bs[int(.975 * reps) - 1]], n_forbidden=len(xa), n_permitted=len(xb))
        o['auroc_forbidden_vs_early'] = dict(nonread=auc_boot(nonread['FORBID'], nonread['EARLY']), follows_read=auc_boot(read['FORBID'], read['EARLY']),
                                             all=auc_boot(by_g['FORBID'], by_g['EARLY']))
        o['auroc_forbidden_vs_late'] = dict(all=auc_boot(by_g['FORBID'], by_g['LATE']), follows_read=auc_boot(read['FORBID'], read['LATE']))
        e15[det] = o
    saved15 = json.loads((E15R / 'results.json').read_text())
    e15['probe_paper_check'] = dict(forbid_nonread=(e15['probe']['nonread']['FORBID']['over'], e15['probe']['nonread']['FORBID']['n'], saved15['primary_nonread']['FORBID']['over'], saved15['primary_nonread']['FORBID']['n']),
                                    early_nonread=(e15['probe']['nonread']['EARLY']['over'], e15['probe']['nonread']['EARLY']['n'], saved15['primary_nonread']['EARLY']['over'], saved15['primary_nonread']['EARLY']['n']),
                                    diff=(e15['probe']['diff_EARLY_minus_FORBID_nonread'], saved15['diff_EARLY_minus_FORBID_nonread']))
    res['E15'] = e15
    print('E15 done', json.dumps(e15['probe_paper_check']), flush=True)

    # ---------------- what does the latest-message view contain on E15 sharing calls?
    role_ct = defaultdict(lambda: defaultdict(int))
    for c in calls:
        if not c['label']: continue
        req, _ = request_for(c['call_id']); last = req['messages'][-1]
        g = 'FORBID' if c['arm'] in ('FORBID', 'FORBID_OLD') else c['arm']
        role_ct[g][last.get('role')] += 1
        role_ct[g]['hedge_regex_in_latest'] += bool(HEDGE.search(json.dumps(last, ensure_ascii=False)))
        asst = ' '.join((m.get('content') or '') for m in req['messages'] if m.get('role') == 'assistant')
        role_ct[g]['hedge_regex_in_earlier_assistant_text'] += bool(HEDGE.search(asst))
        role_ct[g]['n'] += 1
    res['E15_sharing_call_inputs'] = {g: dict(v) for g, v in role_ct.items()}
    (OUT / 'task1_bow.json').write_text(json.dumps(res, indent=1, default=float))
    print('wrote task1_bow.json')


if __name__ == '__main__':
    main()
