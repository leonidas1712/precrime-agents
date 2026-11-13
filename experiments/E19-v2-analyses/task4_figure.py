"""Figure for task 4: test AUROC by layer (95% family-bootstrap CI) and the saved per-seed validation AUROC."""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent
r = json.loads((OUT / 'task4_layers.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, key, title in [(axes[0], 'qwen', 'Qwen3-32B'), (axes[1], 'gptoss', 'gpt-oss-20b')]:
    x = r[key]; L = [int(k) for k in x['test_by_layer']]
    t = [x['test_by_layer'][str(l)]['test_auc'] for l in L]
    lo = [x['test_by_layer'][str(l)]['test_ci95'][0] for l in L]; hi = [x['test_by_layer'][str(l)]['test_ci95'][1] for l in L]
    ax.fill_between(L, lo, hi, color='#999999', alpha=.25, lw=0, label='test 95% CI')
    ax.plot(L, t, 'k-o', ms=3, label=f"test ({x['test_calls']} calls, {x['test_positives']} leaks)")
    for s, c in zip(sorted(x['validation_by_seed']), ['#0072B2', '#D55E00', '#009E73']):
        ax.plot(L, [x['validation_by_seed'][s][str(l)] for l in L], '--', color=c, lw=1, label=f'validation, seed {s}')
    for l in set(x['chosen_layers']):
        ax.axvline(l, color='#CC79A7', lw=.8, ls=':')
    ax.set_title(title + f" (chosen layers {', '.join(map(str, x['chosen_layers']))})", fontsize=9)
    ax.set_xlabel('layer'); ax.set_ylabel('AUROC within scenario x round'); ax.set_ylim(.5, 1.01)
axes[1].legend(fontsize=6.5, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(OUT / 'task4_layers.png', dpi=180); fig.savefig(OUT / 'task4_layers.pdf')
