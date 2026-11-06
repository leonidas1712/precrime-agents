"""Figure: probe scores of the three kinds of scored write in the crossed-destination episodes (E17), one panel per scenario.

Each scenario's scores come from the one probe that held that scenario out, so scores compare within a panel only.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from style import LABEL_SIZE, configure  # noqa: E402
from make_e17_design import ASKA, COMP, FORB  # noqa: E402

RUN = ROOT / 'runs/20261105-1952-e17-probe-analysis'
GROUPS = [('test1_forbidden', 'Forbidden write', FORB),
          ('test1_compromise', 'Compromise', COMP),
          ('test2_pa_allowed', 'Ask-allowed write', ASKA)]
SCENARIOS = [('service-capacity-figure', 'Capacity figure'), ('dispatch-integration-code', 'Dispatch code'),
             ('service-rollout-threshold', 'Rollout threshold'), ('artifact-registry-token', 'Registry token')]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    configure()
    scores = defaultdict(list)
    with open(RUN / 'scores.csv') as f:
        for r in csv.DictReader(f):
            scores[(r['scenario'], r['row'])].append(float(r['probe_score']))
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.45))
    rng = np.random.default_rng(0)
    for ax, (sc, title) in zip(axes, SCENARIOS):
        for k, (row, _, color) in enumerate(GROUPS):
            v = np.array(scores[(sc, row)])
            ax.scatter(k + rng.uniform(-.22, .22, len(v)), v, s=5, color=color, alpha=.45, lw=0, zorder=2)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([k - .3, k + .3], [med, med], color=color, lw=2.2, zorder=3, solid_capstyle='butt')
            ax.plot([k, k], [q1, q3], color=color, lw=1.0, zorder=3)
            ax.text(k, -0.04, f'{len(v)}', transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=LABEL_SIZE - 1.5, color=color)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_xlim(-.6, 2.6)
    axes[0].set_ylabel('Probe score', fontsize=8.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker='o', ms=4, lw=2.2, label=l) for _, l, c in GROUPS]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False, handlelength=1.6,
               columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.09, 1, 1))
    out = ROOT / 'paper-arxiv/figures/fig_e17_scores'
    for ext in ('pdf', 'png'):
        fig.savefig(out.with_suffix('.' + ext), dpi=300, facecolor='white')
    print(out)


if __name__ == '__main__':
    main()
