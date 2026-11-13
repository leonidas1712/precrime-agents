"""Extra checks: message-type-only score; peer-message subgroup (and its drivers); gpt retained-reasoning subgroup."""
import json, pickle
from collections import Counter, defaultdict
import numpy as np
from common import *

o = pickle.loads((CACHE/'fits.pkl').read_bytes())
D = {'gpt': dev('gpt'), 'qwen': dev('qwen')}; F = fresh_qwen()
SETS = {'qwen_dev': ('qwen', D['qwen']['texts']['latest_input'], D['qwen']['rows'], 'y_immediate', o['qwen_loso']['pred'], D['qwen']['probe'], D['qwen']['folds']),
        'qwen_test': ('qwen', F['texts']['latest_input'], F['rows'], 'label', o['qwen_e12']['pred'], np.array([r['probe'] for r in F['rows']])[None], np.zeros((1, len(F['rows'])), int)),
        'gpt_dev': ('gpt', D['gpt']['texts']['full_input'], D['gpt']['rows'], 'y_immediate', o['gpt_loso']['pred'], D['gpt']['probe'], D['gpt']['folds'])}


def ev(rows, y, s, folds, mask=None):
    s = np.atleast_2d(s); s = np.tile(s, (folds.shape[0], 1)) if s.shape[0] == 1 else s
    ids = np.arange(len(rows)) if mask is None else np.flatnonzero(mask)
    if len(set(y[ids])) < 2: return None
    return dict(auc=stream_metrics(rows, ids, y[ids], s[:, ids], folds[:, ids], 1)[0]['within_round_auc'], calls=len(ids), leaking=int(y[ids].sum()))


def contrib(f, text):
    tf = vectorizer(); tf.vocabulary_ = {w: i for i, w in enumerate(f['features'])}; tf.idf_ = f['idf']
    x = tf.transform([text]); return {f['features'][j]: float(v * f['coef'][j]) for j, v in zip(x.indices, x.data)}


out = {}
for name, (model, texts, rows, yk, clf, probe, folds) in SETS.items():
    y = np.array([r[yk] for r in rows])
    lm = [json.loads(t) if model == 'qwen' else json.loads(t)['messages'][-1] for t in texts]
    sec = np.array(['PROTECTED_VALUE' in json.dumps(m) for m in lm]); tool = np.array([m.get('role') == 'tool' for m in lm]); peer = np.array([m.get('role') == 'user' for m in lm])
    mtype = np.where(sec, 2., np.where(peer, 1., 0.))  # secret-bearing tool result > peer message > other tool result
    res = {'message-type-only score (secret-bearing tool result > peer message > other tool result)': ev(rows, y, mtype, folds)}
    if model == 'gpt':
        rs = np.array(['reasoning_details' in t for t in texts])
        res['message type + retained reasoning present (2-feature ordinal: type*2 + reasoning)'] = ev(rows, y, mtype * 2 + rs, folds)
        res['subgroup: input has retained same-turn reasoning'] = dict(text_classifier=ev(rows, y, clf, folds, rs), probe=ev(rows, y, probe, folds, rs))
        res['subgroup: no retained reasoning'] = dict(text_classifier=ev(rows, y, clf, folds, ~rs), probe=ev(rows, y, probe, folds, ~rs))
    res['subgroup: latest message is a peer message'] = dict(text_classifier=ev(rows, y, clf, folds, peer), probe=ev(rows, y, probe, folds, peer))
    res['subgroup: latest message is a tool result'] = dict(text_classifier=ev(rows, y, clf, folds, tool), probe=ev(rows, y, probe, folds, tool))
    res['counts'] = {'leaking after secret-bearing tool result': int((y & sec).sum()), 'leaking after peer message': int((y & peer).sum()), 'leaking after other tool result': int((y & tool & ~sec).sum()),
                     'non-leaking after secret-bearing tool result': int((~y.astype(bool) & sec).sum()), 'non-leaking after peer message': int((~y.astype(bool) & peer).sum()), 'non-leaking after other tool result': int((~y.astype(bool) & tool & ~sec).sum())}
    # drivers within peer-message calls (Qwen): leaking peer-message calls vs peer-message non-leaks in same scenario+round
    if model == 'qwen':
        cell = defaultdict(list)
        for i, r in enumerate(rows):
            if peer[i]: cell[(r['scenario'], r['round'])].append(i)
        def fit_of(i):
            if name == 'qwen_test': return o['qwen_e12']['fit']
            return next(f for f in o['qwen_loso']['fits'] if i in set(f['test']))
        diff = Counter(); used = 0
        for i in np.flatnonzero(peer & (y == 1)):
            neg = [j for j in cell[(rows[i]['scenario'], rows[i]['round'])] if not y[j]]
            if not neg: continue
            used += 1; d = Counter(contrib(fit_of(i), texts[i]))
            for j in neg: d.subtract({k: v / len(neg) for k, v in contrib(fit_of(j), texts[j]).items()})
            diff.update(d)
        res['peer-message drivers'] = dict(leaking_calls=used, raise_leak=[(k, v / used) for k, v in diff.most_common(20)], lower_leak=[(k, v / used) for k, v in sorted(diff.items(), key=lambda t: t[1])[:12]])
        # how tied are scores after secret-bearing reads? distinct scores per scenario+round cell
        sc = np.asarray(clf).ravel(); cells = defaultdict(list)
        for i in np.flatnonzero(sec): cells[(rows[i]['scenario'], rows[i]['round'])].append(round(float(sc[i]), 6))
        res['post-read cells with >1 distinct classifier score'] = f"{sum(len(set(v)) > 1 for v in cells.values())}/{len(cells)}"
    out[name] = res; print(name, json.dumps(res, default=float)[:2500], flush=True)
(OUT/'results_extra.json').write_text(json.dumps(out, indent=1, default=float))
