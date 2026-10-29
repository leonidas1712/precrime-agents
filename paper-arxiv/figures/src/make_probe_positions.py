"""Schematic: which token positions of one model call each detector reads.

Shares the palette and drawing helpers of the setup figure (make_setup.py), so
the two read as one paper. Pure matplotlib patches; no data are plotted.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import make_setup as S
from make_setup import (INK, MUTED, RULE, TARGET, TARGET_BG, TARGET_DK, PEER_BG,
                        PEER_DK, OBS_DK, OBS_BG, ZERO_BG, rbox, eye)

PAPER = Path(__file__).resolve().parents[2]
W, H = 5.5, 4.11
S.W, S.H = W, H          # Canvas reads these module globals

FS = 7.4
FS_SMALL = 7.0
mpl.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})

# One colour per detector (Okabe-Ito; luminance differs, plus a fill pattern each)
LAST = OBS_DK            # last-prompt probe: dark green, as the probe card in Fig. 2
LAST_BG = OBS_BG
INC = '#0072B2'          # incoming-only probe: blue, hatched
INC_BG = '#DCEAF5'
RSN = '#AA4499'          # reasoning probe: purple, dotted fill
RSN_BG = '#F3E3F0'
TXT = '#4A4A4A'          # text detectors: grey, dashed outlines
TXT_BG = '#F2F2F2'
FILT = '#7D7D7D'
HIST_BG = '#EEF1F5'
NEW_BG = '#FFF0C7'
NEW_DK = '#8A5F00'


def main():
    c = S.Canvas()
    ax = c.ax

    # ------------------------------------------------------------ geometry
    lab_x = 0.04
    x0, x1 = 0.04, W - 0.12        # the strip spans the full width
    gap = 0.035
    gen_gap = 0.16                 # room for the "generation starts" line
    segs = [  # key, bold name ('|' breaks lines), description, width (in), fill, edge
        ('sys', 'System prompt', 'role, operator\'s rule, tool definitions', 0.98, ZERO_BG, MUTED),
        ('task', 'Task', 'first user message', 0.52, ZERO_BG, MUTED),
        ('hist', 'History', 'earlier messages; own replies and tool calls; tool results',
         1.15, HIST_BG, MUTED),
        ('peer', 'Latest|peer|message', '', 0.62, NEW_BG, NEW_DK),
        ('tools', 'Tool|results', '', 0.55, NEW_BG, NEW_DK),
        ('z', 'Reasoning|$z_t$', 'gpt-oss-20b only', 0.67, RSN_BG, RSN),
        ('o', 'Output|$o_t$', 'reply or tool call', 0.58, TARGET_BG, TARGET_DK),
    ]
    avail = (x1 - x0) - gap * (len(segs) - 2) - gen_gap
    scale = avail / sum(s[3] for s in segs)
    X = {}
    x = x0
    for key, _n, _d, w, *_ in segs:
        X[key] = (x, x + w * scale)
        x += w * scale + (gen_gap if key == 'tools' else gap)
    gen_x = X['tools'][1] + gen_gap / 2
    prompt_span = (X['sys'][0], X['tools'][1])
    new_span = (X['peer'][0], X['tools'][1])
    end_x = X['o'][1]

    # ------------------------------------------------------------ braces on top
    def brace(xa, xb, y, text, color, size=FS, drop=0.05):
        c.line([xa, xa, xb, xb], [y - drop, y, y, y - drop], color=color, lw=0.8)
        tw = c.width(text.replace('$', ''), size, 'bold') + 0.14
        assert tw < xb - xa, text
        mid = (xa + xb) / 2
        c.add(Rectangle((mid - tw / 2, y - 0.06), tw, 0.12, fc='white', ec='none', zorder=8))
        c.text(mid, y, text, size=size, weight='bold', color=color, ha='center', va='center')

    y_brace = H - 0.10
    brace(prompt_span[0] + 0.02, prompt_span[1] - 0.02, y_brace,
          'Prompt $x_t$: everything the model reads in call $t$', INK)
    brace(X['z'][0] + 0.02, X['o'][1] - 0.02, y_brace, 'Generated in call $t$', TARGET_DK)
    y_new = y_brace - 0.20
    brace(new_span[0] + 0.02, new_span[1] - 0.02, y_new, 'new since last call', NEW_DK,
          size=FS_SMALL, drop=0.04)

    # ------------------------------------------------------------ segments
    seg_top = y_new - 0.13
    seg_h = 0.56
    seg_bot = seg_top - seg_h
    lh = FS_SMALL * 1.18 / 72
    lhb = FS * 1.2 / 72
    for key, name, desc, _w, fc, ec in segs:
        xa, xb = X[key]
        ls = (0, (3, 1.6)) if key == 'z' else '-'
        rbox(c, xa, seg_bot, xb - xa, seg_h, fc, ec, lw=0.8, r=0.045, z=2, ls=ls)
        inner = xb - xa - 0.07
        nl = name.split('|')
        assert all(c.width(ln.replace('$', ''), FS, 'bold') < inner for ln in nl), (key, nl)
        dl = c.wrap(desc, inner, FS_SMALL) if desc else []
        assert all(c.width(ln, FS_SMALL) < inner for ln in dl), (key, dl)
        stack = [(ln, FS, 'bold', ec if ec != MUTED else INK, lhb) for ln in nl]
        stack += [(ln, FS_SMALL, 'normal', MUTED, lh) for ln in dl]
        hgt = sum(st[-1] for st in stack) + (0.03 if dl else 0)
        assert hgt < seg_h - 0.05, (key, hgt)
        yy = (seg_top + seg_bot) / 2 + hgt / 2
        for i, (ln, size, wt, col, step) in enumerate(stack):
            if i == len(nl):
                yy -= 0.03
            c.text((xa + xb) / 2, yy - step / 2, ln, size=size, weight=wt, color=col,
                   ha='center', va='center')
            yy -= step

    # token ruler under the segments: one cell per (schematic) token
    tok_h = 0.085
    tok_y = seg_bot - 0.06 - tok_h
    cell = 0.075
    cells = {}
    for key, *_ in segs:
        xa, xb = X[key]
        n = max(2, int(round((xb - xa) / cell)))
        cw = (xb - xa) / n
        cells[key] = [(xa + i * cw, cw) for i in range(n)]
        fc = {'z': RSN_BG, 'o': TARGET_BG, 'peer': NEW_BG, 'tools': NEW_BG}.get(key, '#E6E9EE')
        for cx, w in cells[key]:
            c.add(Rectangle((cx + 0.007, tok_y), w - 0.014, tok_h, fc=fc, ec='#B9C1CB',
                            lw=0.4, zorder=3))

    # ------------------------------------------------------------ detector rows
    row_h = 0.19
    pitch = 0.37
    head = 0.165                   # header line sits this far above the band
    grp = 0.20                     # extra room for a group caption
    y = tok_y - 0.10 - grp
    rows_y = []
    for i in range(6):
        if i == 3:
            y -= grp
        y -= head + row_h / 2
        rows_y.append(y)
        y -= row_h / 2 + (pitch - head - row_h)
    bottom = rows_y[-1] - row_h / 2 - 0.06

    c.line([gen_x, gen_x], [seg_top + 0.02, bottom], color=INK, lw=0.9,
           ls=(0, (4, 2)), zorder=1)
    c.text(gen_x, bottom - 0.035, 'generation starts', size=FS_SMALL, weight='bold',
           ha='center', va='top')

    def caption(y_top, text):
        yc = y_top + head + row_h / 2 + 0.11
        c.line([lab_x, x1], [yc, yc], color=RULE, lw=0.6, zorder=1)
        t = c.text(lab_x, yc, text, size=FS_SMALL, style='italic', color=MUTED, va='center')
        bb = t.get_window_extent(c.renderer).transformed(c.ax.transData.inverted())
        c.add(Rectangle((lab_x - 0.02, yc - 0.06), bb.x1 - lab_x + 0.08, 0.12, fc='white',
                        ec='none', zorder=8))
        return bb.x1

    def header(y, name, sub, color):
        yh = y + head
        c.text(lab_x, yh, name, size=FS, weight='bold', color=color, va='center')
        c.text(lab_x + c.width(name, FS, 'bold') + 0.08, yh, sub, size=FS_SMALL,
               color=MUTED, va='center')

    def band(xa, xb, y, fc, ec, hatch=None, ls='-', lw=0.9, z=4):
        r = 0.035
        return c.add(FancyBboxPatch((xa + r, y - row_h / 2 + r), xb - xa - 2 * r,
                                    row_h - 2 * r, boxstyle=f'round,pad={r}', fc=fc,
                                    ec=ec, lw=lw, ls=ls, zorder=z, hatch=hatch))

    def ready(x, y, color, marker='o'):
        if marker == 'o':
            c.add(Circle((x, y), 0.045, fc=color, ec='white', lw=0.8, zorder=9))
        else:
            c.add(Rectangle((x - 0.042, y - 0.042), 0.084, 0.084, fc=color, ec='white',
                            lw=0.8, zorder=9))

    def lead(xa, xb, y, color):
        c.line([xa, xb], [y, y], color=color, lw=0.8, ls=(0, (1, 1.5)), zorder=3)

    def note(x, y, s, color, ha='right'):
        return c.text(x, y, s, size=FS_SMALL, color=color, ha=ha, va='center', zorder=10)

    mpl.rcParams['hatch.linewidth'] = 0.5
    cap_end = caption(rows_y[0], 'Probes read activations $h_\\ell \\,\\in\\, \\mathbb{R}^d$ at layer $\\ell$ ($d$ = 5120 Qwen3-32B, 2880 gpt-oss-20b)')
    caption(rows_y[3], 'Text detectors read text, not activations')

    # 1. last-prompt probe: one position, the last prompt token
    y = rows_y[0]
    header(y, 'Last-prompt probe', 'main detector; both models', LAST)
    lx, lw_ = cells['tools'][-1]
    assert cap_end < lx - 0.05, cap_end
    c.add(Rectangle((lx + 0.007, tok_y), lw_ - 0.014, tok_h, fc=LAST, ec=LAST, lw=0.6,
                    zorder=4))
    band(lx - 0.03, lx + lw_ + 0.03, y, LAST_BG, LAST, lw=1.0)
    c.add(Rectangle((lx + 0.007, y - tok_h / 2), lw_ - 0.014, tok_h, fc=LAST, ec='none',
                    zorder=6))
    c.line([lx + lw_ / 2] * 2, [tok_y, y + tok_h / 2], color=LAST, lw=1.2, zorder=3)
    note(lx - 0.07, y, 'last prompt token only:  $s = w^{\\mathrm{T}} h_\\ell + b$', LAST)
    lead(lx + lw_ + 0.03, gen_x - 0.05, y, LAST)
    ready(gen_x, y, LAST)

    # 2. incoming-only probe: every token new since the previous call
    y = rows_y[1]
    header(y, 'Incoming-only probe', 'gpt-oss-20b; exploratory', INC)
    band(*new_span, y, INC_BG, INC, hatch='//////')
    note(new_span[0] - 0.07, y, 'score each new token, average over 16 tokens, take the max', INC)
    lead(new_span[1], gen_x - 0.05, y, INC)
    ready(gen_x, y, INC)

    # 3. reasoning probe: every reasoning token, max
    y = rows_y[2]
    header(y, 'Reasoning probe', 'gpt-oss-20b', RSN)
    band(*X['z'], y, RSN_BG, RSN, hatch='.....')
    lead(gen_x - 0.05, X['z'][0], y, RSN)
    note(gen_x - 0.10, y, '$w^{\\mathrm{T}} h_\\ell + b$ on every reasoning token, take the max', RSN)
    ready(X['z'][1] + gap / 2, y, RSN)

    # 4. zero-shot text monitor
    y = rows_y[3]
    header(y, 'Text monitor (Jev)', 'zero-shot; both models', TXT)
    band(*prompt_span, y, 'white', TXT, ls=(0, (3, 1.6)))
    eye(c, prompt_span[0] + 0.16, y, 0.15, TXT, z=8)
    note(prompt_span[0] + 0.30, y, 'text of $x_t$, secret masked', TXT, ha='left')
    lead(prompt_span[1], gen_x - 0.05, y, TXT)
    ready(gen_x, y, TXT, marker='s')

    # 5. trained text classifier: full input, or latest message only
    y = rows_y[4]
    header(y, 'Text classifier', 'trained bag of words; both models', TXT)
    band(*prompt_span, y, 'white', TXT, ls=(0, (1, 1.3)))
    note(prompt_span[0] + 0.10, y, 'full $x_t$, secret masked', TXT, ha='left')
    note(X['peer'][0] - 0.07, y, 'or the latest message only \u2192', TXT)
    band(*X['peer'], y, TXT_BG, TXT, lw=0.8, z=5)
    lead(prompt_span[1], gen_x - 0.05, y, TXT)
    ready(gen_x, y, TXT, marker='s')

    # 6. output filter, for contrast
    y = rows_y[5]
    header(y, 'Output filter', 'for contrast', FILT)
    band(*X['o'], y, '#F4F4F4', FILT, hatch='xxxx')
    lead(gen_x - 0.05, X['o'][0], y, FILT)
    note(gen_x - 0.10, y, 'matches the secret once it is written', FILT)
    ready(end_x, y, FILT, marker='s')

    # legend for the availability markers
    ly = bottom - 0.09
    lx0 = lab_x + 0.04
    c.add(Circle((lx0, ly), 0.04, fc=INK, ec='none'))
    c.add(Rectangle((lx0 + 0.07, ly - 0.037), 0.074, 0.074, fc=INK, ec='none'))
    c.text(lx0 + 0.20, ly, 'score available (circle: probe, square: text)', size=FS_SMALL,
           color=INK, va='center')
    assert ly - 0.06 > 0.03, ly  # nothing cut off at the bottom

    out = PAPER / 'figures/fig_probe_positions'
    c.fig.savefig(out.with_suffix('.pdf'), facecolor='white')
    c.fig.savefig(out.with_suffix('.png'), dpi=200, facecolor='white')
    out.with_suffix('.json').write_text(json.dumps({
        'source': 'paper-arxiv/figures/src/make_probe_positions.py',
        'width_inches': W, 'height_inches': H,
        'note': 'schematic of one model call: token segments of x_t, z_t, o_t and the '
                'positions each detector reads; token counts are illustrative, not data'},
        indent=2) + '\n')
    plt.close(c.fig)


if __name__ == '__main__':
    main()
