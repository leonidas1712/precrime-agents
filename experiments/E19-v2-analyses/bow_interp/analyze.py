"""Steps 2-5: weights + stability, structural single-cue baselines, per-call contributions, ablations, subgroups."""
import csv, json, pickle, re
from collections import Counter, defaultdict
import numpy as np
from common import *

o = pickle.loads((CACHE/'fits.pkl').read_bytes())
R = {}  # everything that goes to results.json
N = 30

# ---------------------------------------------------------------- data handles
D = {'gpt': dev('gpt'), 'qwen': dev('qwen')}
VIEW = {'gpt': 'full_input', 'qwen': 'latest_input'}
F = fresh_qwen()
for r, s in zip(F['rows'], o['qwen_e12']['pred']): r['bow'] = float(s)


def latest_msg(model, t):
    return json.loads(t) if model == 'qwen' else json.loads(t)['messages'][-1]


def ev(kind, scores, subset=None):
    """Paper metric on the paper's calls. kind: 'gpt'/'qwen' dev (LOSO folds) or 'test' (Qwen test)."""
    if kind == 'test':
        rows = F['rows']; y = np.array([r['label'] for r in rows]); folds = np.zeros((1, len(rows)), int); s = np.atleast_2d(scores)
    else:
        rows = D[kind]['rows']; y = np.array([r['y_immediate'] for r in rows]); folds = D[kind]['folds']; s = np.atleast_2d(scores)
        if s.shape[0] == 1: s = np.tile(s, (3, 1))
        if folds.shape[0] != s.shape[0]: folds = folds[:s.shape[0]]
    ids = np.arange(len(rows)) if subset is None else np.flatnonzero(subset)
    if len(set(y[ids])) < 2: return None
    m = stream_metrics(rows, ids, y[ids], s[:, ids], folds[:, ids], 1)[0]
    return dict(auc=m['within_round_auc'], calls=len(ids), leaking=int(y[ids].sum()))


# ---------------------------------------------------------------- 2. weights and stability
def top(features, coef, n, sign):
    i = np.argsort(coef if sign < 0 else -coef)[:n]; return [(features[j], float(coef[j])) for j in i]


def fold_table(fits):
    allf = sorted(set().union(*[set(f['features']) for f in fits])); idx = {w: i for i, w in enumerate(allf)}
    M = np.zeros((len(fits), len(allf)))
    for k, f in enumerate(fits): M[k, [idx[w] for w in f['features']]] = f['coef']
    return np.array(allf), M


def stability(fits, ref_feats, ref_coef, n=N, k20=20):
    feats, M = fold_table(fits); idx = {w: i for i, w in enumerate(feats)}
    tops = {s: [set(w for w, _ in top(f['features'], f['coef'], n, s)) for f in fits] for s in (1, -1)}
    tops20 = {s: [set(w for w, _ in top(f['features'], f['coef'], k20, s)) for f in fits] for s in (1, -1)}
    out = {}
    for s, name in [(1, 'positive'), (-1, 'negative')]:
        table = []
        for w, c in top(ref_feats, ref_coef, n, s):
            j = idx.get(w)
            col = M[:, j] if j is not None else np.zeros(len(fits))
            table.append(dict(feature=w, weight=c, folds_in_top30=sum(w in t for t in tops[s]), folds_same_sign=int(np.sum(np.sign(col) == s)),
                              fold_mean=float(col.mean()), fold_min=float(col.min()), fold_max=float(col.max())))
        ref20 = [w for w, _ in top(ref_feats, ref_coef, k20, s)]
        out[name] = dict(table=table, top20_in_majority_of_fold_top20=sum(sum(w in t for t in tops20[s]) >= (len(fits) // 2 + 1) for w in ref20),
                         mean_pairwise_jaccard_top20=float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(tops20[s]) for b in tops20[s][i + 1:]])))
    return out


feats, M = fold_table(o['gpt_loso']['fits']); gmean = M.mean(0)
R['weights'] = {
    'qwen_test_classifier (all 2976 dev calls, latest message; stability over 6 LOSO dev fits)': stability(o['qwen_loso']['fits'], o['qwen_e12']['fit']['features'], o['qwen_e12']['fit']['coef']),
    'gpt_dev_classifier (mean of 6 LOSO fits, missing=0, full input)': stability(o['gpt_loso']['fits'], feats, gmean),
}
# Also: qwen LOSO mean weights for reference
qf, QM = fold_table(o['qwen_loso']['fits'])
R['weights_qwen_loso_mean_top'] = dict(positive=top(qf, QM.mean(0), N, 1), negative=top(qf, QM.mean(0), N, -1))
ga = o['gpt_alldev']; ai = {w: i for i, w in enumerate(ga['features'])}
for t in R['weights']['gpt_dev_classifier (mean of 6 LOSO fits, missing=0, full input)'].values():
    for row in t['table']: row['all_dev_fit_weight'] = float(ga['coef'][ai[row['feature']]]) if row['feature'] in ai else None
# full CSVs
with open(OUT/'weights_qwen_test_classifier.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['feature', 'weight', 'idf'])
    ff = o['qwen_e12']['fit']; [w.writerow([a, f'{b:.5f}', f'{c:.4f}']) for a, b, c in sorted(zip(ff['features'], ff['coef'], ff['idf']), key=lambda t: -abs(t[1]))]
with open(OUT/'weights_gpt_fold_mean.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['feature', 'fold_mean_weight'] + [f'fold{k}' for k in range(len(M))])
    for j in np.argsort(-np.abs(gmean)): w.writerow([feats[j], f'{gmean[j]:.5f}'] + [f'{v:.5f}' for v in M[:, j]])

# ---------------------------------------------------------------- 4a. structural single cues (same calls, same metric)
def cues(model, texts, rows):
    lm = [latest_msg(model, t) for t in texts]
    c = {'latest message contains masked secret (PROTECTED_VALUE)': [('PROTECTED_VALUE' in json.dumps(m)) for m in lm],
         'latest message is a tool result': [m.get('role') == 'tool' for m in lm],
         'latest message is a send/post confirmation (delivered: true)': [('delivered' in json.dumps(m)) for m in lm]}
    if model == 'gpt':
        c['input contains retained same-turn reasoning (reasoning_details field)'] = ['reasoning_details' in t for t in texts]
        c['masked secret anywhere in input'] = ['PROTECTED_VALUE' in t for t in texts]
    return {k: np.array(v, float) for k, v in c.items()}


R['cues'] = {}
for kind, texts, rows, ykey in [('qwen', D['qwen']['texts']['latest_input'], D['qwen']['rows'], 'y_immediate'), ('test', F['texts']['latest_input'], F['rows'], 'label'), ('gpt', D['gpt']['texts']['full_input'], D['gpt']['rows'], 'y_immediate')]:
    model = 'gpt' if kind == 'gpt' else 'qwen'; y = np.array([r[ykey] for r in rows]); C = cues(model, texts, rows); R['cues'][kind] = {}
    for name, v in C.items():
        R['cues'][kind][name] = dict(auc=ev(kind, v)['auc'], share_of_leaking=float(v[y == 1].mean()), share_of_nonleaking=float(v[y == 0].mean()))
    pv = C['latest message contains masked secret (PROTECTED_VALUE)'] > 0
    clf = o['qwen_e12']['pred'] if kind == 'test' else o[f'{model}_loso']['pred']
    probe = np.array([r['probe'] for r in rows]) if kind == 'test' else D[model]['probe']
    R['cues'][kind]['subgroups'] = {sub: dict(text_classifier=ev(kind, clf, m), probe=ev(kind, probe, m)) for sub, m in
                                    [('all calls', np.ones(len(y), bool)), ('latest message does NOT contain masked secret', ~pv), ('latest message contains masked secret', pv)]}
    R['cues'][kind]['classifier_plus_cue_check'] = 'see subgroups'
    print(kind, json.dumps(R['cues'][kind], default=float)[:1500], flush=True)

# ---------------------------------------------------------------- 3. per-call contributions
def contrib_vec(tf_feats, coef, idf, text):
    tf = vectorizer(); tf.vocabulary_ = {w: i for i, w in enumerate(tf_feats)}; tf.idf_ = idf
    x = tf.transform([text]); return {tf_feats[j]: float(v * coef[j]) for j, v in zip(x.indices, x.data)}


def fit_of(kind, i):
    if kind == 'test': f = o['qwen_e12']['fit']; return f
    for f in o[f'{kind}_loso']['fits']:
        if i in set(f['test']): return f


def excerpt(kind, text, words, n=230):
    model = 'gpt' if kind == 'gpt' else 'qwen'; m = latest_msg(model, text)
    s = str(m.get('content') or '')
    s = re.sub(r'\s+', ' ', s)
    hit = [s.lower().find(w.split()[0]) for w in words if s.lower().find(w.split()[0]) >= 0]
    start = max(0, min(hit) - 60) if hit else 0
    out = f"[{m.get('role')}] " + ('…' if start else '') + s[start:start + n] + ('…' if len(s) > start + n else '')
    if model == 'gpt':
        rs = [x.get('reasoning') for x in json.loads(text)['messages'] if x.get('reasoning')]
        if rs: out += ' || retained reasoning: "' + re.sub(r'\s+', ' ', rs[-1])[:160] + '…"'
    return out


R['contributions'] = {}; ILL = []
for kind, texts, rows, ykey in [('test', F['texts']['latest_input'], F['rows'], 'label'), ('gpt', D['gpt']['texts']['full_input'], D['gpt']['rows'], 'y_immediate'), ('qwen', D['qwen']['texts']['latest_input'], D['qwen']['rows'], 'y_immediate')]:
    y = np.array([r[ykey] for r in rows]); scores = o['qwen_e12']['pred'] if kind == 'test' else o[f"{kind}_loso"]['pred']
    cell = defaultdict(list)
    for i, r in enumerate(rows): cell[(r['scenario'], r['round'])].append(i)
    pos = [i for i in range(len(rows)) if y[i]]; diff = Counter(); mass_pos = Counter(); cache = {}
    def cv(i):
        if i not in cache: f = fit_of(kind, i); cache[i] = contrib_vec(f['features'], f['coef'], f['idf'], texts[i])
        return cache[i]
    used = 0
    for i in pos:
        neg = [j for j in cell[(rows[i]['scenario'], rows[i]['round'])] if not y[j]]
        if not neg: continue
        used += 1; ci = cv(i); mass_pos.update(ci)
        nm = Counter()
        for j in neg: nm.update({k: v / len(neg) for k, v in cv(j).items()})
        d = Counter(ci); d.subtract(nm); diff.update(d)
    R['contributions'][kind] = dict(leaking_calls_with_matched_negatives=used,
        top_features_raising_leaks_over_matched=[(k, v / used) for k, v in diff.most_common(25)],
        top_features_lowering_leaks_vs_matched=[(k, v / used) for k, v in sorted(diff.items(), key=lambda t: t[1])[:15]],
        top_mean_contribution_in_leaking=[(k, v / used) for k, v in mass_pos.most_common(15)])
    # illustrative: leaking calls (highest, median, lowest scored) + one matched non-leaking each (highest-scored in same cell)
    if kind == 'qwen': continue
    order = sorted(pos, key=lambda i: -scores[i]); picks = [order[0], order[len(order) // 2], order[-1]]
    if kind == 'test':  # add one leaking call whose latest message is the peer, if not already present
        peer = [i for i in order if json.loads(texts[i])['role'] == 'user' and i not in picks]
        if peer: picks.append(peer[0])
    for i in picks:
        neg = [j for j in cell[(rows[i]['scenario'], rows[i]['round'])] if not y[j]]
        for j, tag in [(i, 'LEAK'), (max(neg, key=lambda j: scores[j]) if neg else None, 'matched non-leak (highest-scored in same scenario+round)')]:
            if j is None: continue
            c = cv(j); upw = sorted(c.items(), key=lambda t: -t[1])[:6]; dnw = sorted(c.items(), key=lambda t: t[1])[:4]
            r = rows[j]
            within = [k for k in cell[(r['scenario'], r['round'])]]
            ILL.append(dict(set={'test': 'Qwen3-32B test (latest message)', 'gpt': 'gpt-oss-20b dev (full input)'}[kind], tag=tag, row_id=r['row_id'], scenario=r['scenario'], round=int(r['round']), label=int(y[j]),
                            score=float(scores[j]), rank_in_cell=f"{1 + sum(scores[k] > scores[j] for k in within)}/{len(within)}",
                            up=[(k, round(v, 3)) for k, v in upw], down=[(k, round(v, 3)) for k, v in dnw], excerpt=excerpt(kind, texts[j], [k for k, _ in upw])))
R['illustrative'] = ILL

# ---------------------------------------------------------------- 5. ablations (refit with the same pipeline)
def topk_words(feats, coef, k, positive_only=False):
    order = np.argsort(-coef if positive_only else -np.abs(coef)); words = []
    for j in order:
        for t in feats[j].split():
            if t not in words: words.append(t)
            if len(words) >= k: return words
    return words


def scen_specific_words(model, train_ids, thresh=.9, min_df=5):
    rows = D[model]['rows']; texts = D[model]['texts'][VIEW[model]]; seen = defaultdict(Counter)
    tok = re.compile(r'(?u)\b\w\w+\b')
    for i in train_ids:
        for t in set(tok.findall(texts[i].lower())): seen[t][rows[i]['scenario']] += 1
    return sorted(t for t, c in seen.items() if sum(c.values()) >= min_df and max(c.values()) / sum(c.values()) >= thresh)


def strip_reasoning(t):
    j = json.loads(t)
    for m in j['messages']: m.pop('reasoning', None); m.pop('reasoning_details', None)
    return json.dumps(j, ensure_ascii=False, sort_keys=True)


def loso_variant(model, stop=None, transform=None, vocab_filter=None):
    d = D[model]; rows = d['rows']; docs = d['texts'][VIEW[model]]
    if transform: docs = [transform(t) for t in docs]
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows]); pred = np.full(len(rows), np.nan); removed = []
    for f0 in o[f'{model}_loso']['fits']:
        ids, te = f0['train'], f0['test']; w = position_weights(y[ids], rounds[ids])  # identical ids/weights as the paper fit
        w = w  # ids already filtered to w>0
        sw = stop(f0, ids) if stop else None; removed.append(len(sw) if sw else 0)
        kw = dict(stop_words=sw) if sw else {}
        if vocab_filter: kw = dict(vocabulary=[v for v in f0['features'] if vocab_filter(v)])
        tf, clf = fit([docs[i] for i in ids], y[ids], position_weights(y[ids], rounds[ids]), **kw)
        pred[te] = clf.decision_function(tf.transform([docs[i] for i in te]))
    return pred, removed


def test_variant(stop=None, vocab_filter=None):
    d = D['qwen']; rows = d['rows']; docs = d['texts']['latest_input']; f0 = o['qwen_e12']['fit']
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows]); ids = f0['train']
    sw = stop(f0, ids) if stop else None; kw = dict(stop_words=sw) if sw else {}
    if vocab_filter: kw = dict(vocabulary=[v for v in f0['features'] if vocab_filter(v)])
    tf, clf = fit([docs[i] for i in ids], y[ids], position_weights(y[ids], rounds[ids]), **kw)
    return clf.decision_function(tf.transform(F['texts']['latest_input'])), [len(sw) if sw else 0]


import warnings; warnings.filterwarnings('ignore', message='Your stop_words may be inconsistent')
abl = {
    'paper classifier (no change)': dict(),
    'remove top-10 words by |weight|': dict(stop=lambda f, ids: topk_words(f['features'], f['coef'], 10)),
    'remove top-50 words by |weight|': dict(stop=lambda f, ids: topk_words(f['features'], f['coef'], 50)),
    'remove top-10 positive-weight words': dict(stop=lambda f, ids: topk_words(f['features'], f['coef'], 10, True)),
    'remove top-50 positive-weight words': dict(stop=lambda f, ids: topk_words(f['features'], f['coef'], 50, True)),
    'remove mask token only (protected_value)': dict(stop=lambda f, ids: ['protected_value']),
}
R['ablations'] = {}
for name, a in abl.items():
    res = {}
    for kind in ['qwen', 'gpt']:
        p, n = loso_variant(kind, stop=a.get('stop')); res[f'{kind}_dev'] = dict(**ev(kind, p), words_removed_per_fit=n)
        if name == 'paper classifier (no change)': assert np.allclose(p, o[f'{kind}_loso']['pred'])
        if 'top-10 words by' in name: res[f'{kind}_dev']['example_removed_fold0'] = a['stop'](o[f'{kind}_loso']['fits'][0], None)
    p, n = test_variant(stop=a.get('stop')); res['qwen_test'] = dict(**ev('test', p), words_removed_per_fit=n)
    if 'top-' in name: res['qwen_test']['removed'] = a['stop'](o['qwen_e12']['fit'], None)
    R['ablations'][name] = res; print('ABL', name, {k: round(v['auc'], 3) for k, v in res.items()}, flush=True)

# scenario-specific words (>=90% of training docs containing the word come from one scenario), from each fit's own training calls
res = {}
for kind in ['qwen', 'gpt']:
    p, n = loso_variant(kind, stop=lambda f, ids, kind=kind: scen_specific_words(kind, ids)); res[f'{kind}_dev'] = dict(**ev(kind, p), words_removed_per_fit=n)
p, n = test_variant(stop=lambda f, ids: scen_specific_words('qwen', ids)); res['qwen_test'] = dict(**ev('test', p), words_removed_per_fit=n)
R['ablations']['remove scenario-specific words (>=90% of docs in one scenario, df>=5)'] = res; print('ABL scen', {k: round(v['auc'], 3) for k, v in res.items()})

# theme-restricted / structural variants
p, _ = loso_variant('gpt', transform=strip_reasoning)
R['ablations']['gpt: drop retained same-turn reasoning from the input (fields and text)'] = {'gpt_dev': ev('gpt', p)}
RF = {'reasoning', 'reasoning_details', 'text', 'format', 'unknown', 'index', 'type'}
p, _ = loso_variant('gpt', stop=lambda f, ids: sorted(RF))
R['ablations']['gpt: remove reasoning-field tokens only (reasoning, reasoning_details, text, format, unknown, index, type)'] = {'gpt_dev': ev('gpt', p)}
# mask+file-content only: features that occur in tool-result latest messages containing the masked secret only
print('ABL struct', {k: v for k, v in R['ablations'].items() if k.startswith('gpt:')})
(OUT/'results.json').write_text(json.dumps(R, indent=1, default=float))
print('done')
