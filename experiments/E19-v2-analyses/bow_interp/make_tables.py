"""Render markdown tables from results.json / results_extra.json (pure stdlib)."""
import json
from pathlib import Path
OUT = Path(__file__).parent; R = json.loads((OUT/'results.json').read_text()); X = json.loads((OUT/'results_extra.json').read_text())
L = []
def esc(s): return s.replace('|', '\\|')
for key, v in R['weights'].items():
    gpt = key.startswith('gpt')
    for sign in ['positive', 'negative']:
        t = v[sign]
        L.append(f"\n#### {key}: top 30 {sign}-weight features\n")
        L.append(f"Top-20 overlap: {t['top20_in_majority_of_fold_top20']}/20 of these top-20 are in the top-20 of at least 4 of the 6 LOSO fits; mean pairwise Jaccard of the 6 fold top-20 lists = {t['mean_pairwise_jaccard_top20']:.2f}.\n")
        hdr = '| # | feature | weight | folds in top-30 (of 6) | folds same sign | fold min..max |' + (' all-dev fit |' if gpt else '')
        L.append(hdr); L.append('|' + '---|' * (hdr.count('|') - 1))
        for i, r in enumerate(t['table'], 1):
            row = f"| {i} | `{esc(r['feature'])}` | {r['weight']:+.3f} | {r['folds_in_top30']} | {r['folds_same_sign']} | {r['fold_min']:+.2f}..{r['fold_max']:+.2f} |"
            if gpt: row += f" {r['all_dev_fit_weight']:+.3f} |" if r['all_dev_fit_weight'] is not None else ' – |'
            L.append(row)
L.append('\n#### Ablations (within scenario-and-round AUROC, same calls, refit with the paper recipe)\n')
L.append('| Variant | Qwen3-32B test (latest) | Qwen3-32B dev (latest) | gpt-oss-20b dev (full) |'); L.append('|---|---:|---:|---:|')
for k, v in R['ablations'].items():
    g = lambda n: f"{v[n]['auc']:.3f}" if n in v else '–'
    L.append(f"| {k} | {g('qwen_test')} | {g('qwen_dev')} | {g('gpt_dev')} |")
L.append('\n#### Single structural cues and subgroups (paper metric; scores from the paper classifier/probe, no refit)\n')
L.append('| Cue / subgroup | Qwen3-32B test | Qwen3-32B dev | gpt-oss-20b dev |'); L.append('|---|---:|---:|---:|')
names = list(R['cues']['gpt'].keys())
for n in names:
    if n in ('subgroups', 'classifier_plus_cue_check'): continue
    cell = lambda k: (f"{R['cues'][k][n]['auc']:.3f} ({R['cues'][k][n]['share_of_leaking']:.0%} of leaks vs {R['cues'][k][n]['share_of_nonleaking']:.1%} of non-leaks)" if n in R['cues'][k] else '–')
    L.append(f"| cue alone: {n} | {cell('test')} | {cell('qwen')} | {cell('gpt')} |")
mt = 'message-type-only score (secret-bearing tool result > peer message > other tool result)'
L.append(f"| cue alone: {mt} | {X['qwen_test'][mt]['auc']:.3f} | {X['qwen_dev'][mt]['auc']:.3f} | {X['gpt_dev'][mt]['auc']:.3f} |")
def sub(src, n):
    s = src.get(n);
    if not s: return '–'
    a, b = s['text_classifier'], s['probe']
    return f"text {a['auc']:.3f} / probe {b['auc']:.3f} (n={a['calls']}, {a['leaking']} leaks)" if a and b else '–'
for n in ['all calls', 'latest message does NOT contain masked secret', 'latest message contains masked secret']:
    L.append(f"| subgroup: {n} | {sub(R['cues']['test']['subgroups'], n)} | {sub(R['cues']['qwen']['subgroups'], n)} | {sub(R['cues']['gpt']['subgroups'], n)} |")
for n in ['subgroup: latest message is a peer message', 'subgroup: latest message is a tool result', 'subgroup: input has retained same-turn reasoning', 'subgroup: no retained reasoning']:
    L.append(f"| {n} | {sub(X['qwen_test'], n)} | {sub(X['qwen_dev'], n)} | {sub(X['gpt_dev'], n)} |")
L.append('\n#### Illustrative calls\n')
for i, x in enumerate(R['illustrative'], 1):
    up = ', '.join(f"`{esc(k)}` {v:+.2f}" for k, v in x['up'][:5]); dn = ', '.join(f"`{esc(k)}` {v:+.2f}" for k, v in x['down'][:3])
    L.append(f"{i}. **{x['set']} — {x['tag']}** ({x['scenario']}, round {x['round']}; score {x['score']:+.2f}, rank {x['rank_in_cell']} in its scenario+round)  \n   Excerpt: {esc(x['excerpt'][:330])}  \n   Pushes up: {up}. Pushes down: {dn}.\n")
L.append('\n#### Mean contribution difference, leaking call minus matched non-leaking calls (same scenario and round)\n')
for k, v in R['contributions'].items():
    L.append(f"- **{k}** ({v['leaking_calls_with_matched_negatives']} leaking calls): " + ', '.join(f"`{a}` {b:+.3f}" for a, b in v['top_features_raising_leaks_over_matched'][:15]))
for k in ['qwen_test', 'qwen_dev']:
    v = X[k]['peer-message drivers']
    L.append(f"- **{k}, peer-message calls only** ({v['leaking_calls']} leaking calls): " + ', '.join(f"`{a}` {b:+.3f}" for a, b in v['raise_leak'][:15]))
(OUT/'tables.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L)[:200])
