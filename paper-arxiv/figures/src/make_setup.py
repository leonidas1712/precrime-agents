"""Figure 2: how an episode works (chat excerpt, actors and observers, labels).

The quoted messages are read verbatim from the EXCHANGE constant of the sprint
figure script, so the shortened quotes and the token truncation stay exact.
Everything is drawn with matplotlib patches; no images or emoji fonts.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, PathPatch, Polygon, Rectangle)
from matplotlib.path import Path as MPath

from style import MODEL_COLORS

PAPER = Path(__file__).resolve().parents[2]
SOURCE = PAPER.parent / 'report/figures/make_figure1.py'

W, H = 5.5, 4.2  # inches; included at \linewidth (5.5in)

# Palette (Okabe-Ito based, colour-blind safe; luminance differs for grayscale)
INK = '#1F2933'
MUTED = '#5F6B7A'
RULE = '#C9D1DA'
TARGET = MODEL_COLORS['qwen']      # the example episode is Qwen3-32B
TARGET_BG = '#E4EFF8'
TARGET_DK = '#004F7D'
PEER = '#B8433A'
PEER_BG = '#F8E5E1'
PEER_DK = '#7E2620'
LEAK = '#E69F00'
LEAK_BG = '#FFF0C7'
LEAK_DK = '#8A5F00'
OBS = '#009E73'
OBS_BG = '#DDF2EB'
OBS_DK = '#00674B'
GOLD = '#E0A800'
PAPER_BG = '#FBF6EA'
ZERO_BG = '#F7F8FA'
SCORED_BG = '#EEF3F8'

FS = 7.4      # body text (pt)
FS_SMALL = 7.0
FS_TITLE = 8.6

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': None,
    'mathtext.fontset': 'dejavusans',
})


# ---------------------------------------------------------------- helpers
class Canvas:
    """One axes in inch coordinates covering the whole figure."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis('off')
        self.renderer = self.fig.canvas.get_renderer()

    def width(self, s, size=FS, weight='normal', style='normal'):
        fp = FontProperties(family=mpl.rcParams['font.sans-serif'], size=size,
                            weight=weight, style=style)
        w, _, _ = self.renderer.get_text_width_height_descent(s, fp, ismath=False)
        return w / self.fig.dpi

    def wrap(self, text, width, size=FS):
        lines, cur = [], ''
        for word in text.split(' '):
            trial = (cur + ' ' + word).strip()
            if cur and self.width(trial, size) > width:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        return lines

    def text(self, x, y, s, size=FS, **kw):
        kw.setdefault('color', INK)
        kw.setdefault('zorder', 10)
        return self.ax.text(x, y, s, fontsize=size, **kw)

    def line(self, xs, ys, **kw):
        kw.setdefault('solid_capstyle', 'round')
        return self.ax.plot(xs, ys, **kw)

    def add(self, patch):
        self.ax.add_patch(patch)
        return patch


def rbox(c, x, y, w, h, fc, ec, lw=0.8, r=0.06, z=1, ls='-'):
    return c.add(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                                boxstyle=f'round,pad={r}', fc=fc, ec=ec, lw=lw,
                                zorder=z, linestyle=ls))


def arrow(c, p0, p1, color=INK, lw=0.9, ls='-', rad=0.0, head=6, z=5):
    return c.add(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=head,
                                 lw=lw, color=color, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0,
                                 connectionstyle=f'arc3,rad={rad}',
                                 capstyle='round', joinstyle='round'))


# ---------------------------------------------------------------- icons
def robot(c, cx, cy, s, color, bg, dark, kind='target', body=True, z=6):
    """Friendly robot. (cx, cy) is the centre of the head; s the head width."""
    hw, hh = s, 0.78 * s
    if body:
        bw, bh = 0.74 * s, 0.42 * s
        rbox(c, cx - bw / 2, cy - hh / 2 - bh - 0.04 * s, bw, bh, bg, dark,
             lw=0.8, r=0.12 * s, z=z)
        c.add(Circle((cx, cy - hh / 2 - bh / 2 - 0.04 * s), 0.07 * s, fc=color,
                     ec='none', zorder=z + 1))
    for sx in (-1, 1):  # ears
        ex = cx + sx * hw / 2 - (0.045 * s)
        c.add(Rectangle((ex + sx * 0.05 * s, cy - 0.13 * s), 0.09 * s, 0.26 * s,
                        fc=dark, ec='none', zorder=z))
    rbox(c, cx - hw / 2, cy - hh / 2, hw, hh, bg, dark, lw=0.9, r=0.16 * s, z=z + 1)
    fw, fh = 0.72 * s, 0.46 * s  # face plate
    rbox(c, cx - fw / 2, cy - fh / 2 - 0.02 * s, fw, fh, 'white', dark, lw=0.6,
         r=0.12 * s, z=z + 2)
    for sx in (-1, 1):
        c.add(Circle((cx + sx * 0.17 * s, cy + 0.01 * s), 0.065 * s, fc=dark, ec='none',
                     zorder=z + 3))
    ty = cy + hh / 2
    if kind == 'peer':
        for sx in (-1, 1):  # slanted brows
            c.line([cx + sx * 0.08 * s, cx + sx * 0.27 * s],
                   [cy + 0.09 * s, cy + 0.16 * s], color=dark, lw=0.9, zorder=z + 3)
        c.add(Arc((cx + 0.04 * s, cy - 0.12 * s), 0.22 * s, 0.10 * s, theta1=200,
                  theta2=340, color=dark, lw=0.8, zorder=z + 3))
        for sx in (-1, 1):  # horns
            bx = cx + sx * 0.28 * s
            c.add(Polygon([(bx - 0.10 * s, ty - 0.03 * s), (bx + 0.10 * s, ty - 0.03 * s),
                           (bx + sx * 0.16 * s, ty + 0.30 * s)], closed=True,
                          fc=color, ec=dark, lw=0.6, joinstyle='round', zorder=z))
    else:
        c.add(Arc((cx, cy - 0.10 * s), 0.26 * s, 0.14 * s, theta1=200, theta2=340,
                  color=dark, lw=0.8, zorder=z + 3))
        c.line([cx, cx], [ty, ty + 0.12 * s], color=dark, lw=0.8, zorder=z)
        c.add(Circle((cx, ty + 0.14 * s), 0.05 * s, fc=color, ec='none', zorder=z))
        c.add(Ellipse((cx, ty + 0.30 * s), 0.70 * s, 0.17 * s, fc='none', ec=GOLD,
                      lw=1.3, zorder=z + 4))


def eye(c, cx, cy, s, color, z=7):
    w, h = s / 2, 0.30 * s
    verts = [(cx - w, cy), (cx, cy + 1.8 * h), (cx + w, cy),
             (cx, cy - 1.8 * h), (cx - w, cy)]
    codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3]
    c.add(PathPatch(MPath(verts, codes), fc='white', ec=color, lw=1.0, zorder=z))
    c.add(Circle((cx, cy), 0.17 * s, fc=color, ec='none', zorder=z + 1))
    c.add(Circle((cx + 0.05 * s, cy + 0.05 * s), 0.05 * s, fc='white', ec='none', zorder=z + 2))


def document(c, x, y, w, h, fc, ec, lines=3, z=7):
    f = 0.28 * w
    c.add(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)],
                  closed=True, fc=fc, ec=ec, lw=0.7, joinstyle='round', zorder=z))
    c.add(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)],
                  closed=True, fc=ec, ec=ec, lw=0.5, joinstyle='round', zorder=z + 1))
    for i in range(lines):
        ly = y + h - f - 0.12 * h - i * 0.17 * h
        c.line([x + 0.16 * w, x + w - 0.22 * w], [ly, ly], color=ec, lw=0.6,
               alpha=0.6, zorder=z + 1)


def lock(c, cx, cy, s, color, z=9):
    bw, bh = s, 0.78 * s
    c.add(Arc((cx, cy + bh / 2), 0.62 * s, 0.9 * s, theta1=0, theta2=180, color=color,
              lw=1.1, zorder=z))
    rbox(c, cx - bw / 2, cy - bh / 2, bw, bh, color, color, lw=0.5, r=0.12 * s, z=z + 1)
    c.add(Circle((cx, cy + 0.06 * s), 0.11 * s, fc='white', ec='none', zorder=z + 2))
    c.add(Rectangle((cx - 0.04 * s, cy - 0.22 * s), 0.08 * s, 0.26 * s, fc='white',
                    ec='none', zorder=z + 2))


def scroll(c, x, y, w, h, fc, ec, z=7):
    r = 0.075 * h
    c.add(Rectangle((x, y + r), w, h - 2 * r, fc=fc, ec=ec, lw=0.7, zorder=z))
    for yy in (y + h - r, y + r):
        rbox(c, x - 0.10 * w, yy - r, w * 1.2, 2 * r, fc, ec, lw=0.7, r=r * 0.9, z=z + 1)
    for i in range(3):
        ly = y + h - 3.4 * r - i * 0.17 * h
        c.line([x + 0.18 * w, x + w - 0.18 * w], [ly, ly], color=ec, lw=0.6,
               alpha=0.7, zorder=z + 1)


def needle(c, base, tip, color, z=8, size=1.0):
    """A probe needle from base to its point at tip, with a round grip."""
    base, tip = np.array(base), np.array(tip)
    d = tip - base
    L = np.hypot(*d)
    u, n = d / L, np.array([-d[1], d[0]]) / L
    se = base + u * (L * 0.70)
    hw = 0.026 * size
    c.add(Polygon([base + n * hw, se + n * hw, se - n * hw, base - n * hw], closed=True,
                  fc=color, ec=color, lw=0.4, joinstyle='round', zorder=z))
    c.add(Polygon([se + n * hw * 0.5, tip, se - n * hw * 0.5], closed=True,
                  fc=color, ec=color, lw=0.4, joinstyle='round', zorder=z))
    c.add(Circle(tuple(base), 0.045 * size, fc='white', ec=color, lw=1.0, zorder=z + 1))


def activations(c, x, y, cell, n=5, z=8):
    shades = ['#0072B2', '#A9CBE6', '#3E8EC4', '#D7E7F3', '#6AA6D3']
    for i in range(n):
        c.add(Rectangle((x, y + i * cell), cell, cell, fc=shades[i % len(shades)],
                        ec='white', lw=0.5, zorder=z))
    c.add(Rectangle((x, y), cell, n * cell, fc='none', ec=TARGET_DK, lw=0.6, zorder=z + 1))


def panel_title(c, x, y, letter, title):
    c.text(x, y, f'({letter})  {title}', size=FS_TITLE, weight='bold', va='baseline')


# ---------------------------------------------------------------- panel (a)
def panel_a(c, exchange, x0, x1, ytop):
    panel_title(c, x0, ytop, 'a', 'A real episode: refusal, then disclosure')
    av = 0.30           # avatar head width
    gap = 0.09
    bub_w = (x1 - x0) - av - gap
    pad = 0.062
    lh = FS * 1.2 / 72
    y = ytop - 0.12
    for kind, who, text, note in exchange:
        is_peer, leak = kind == 'peer', kind == 'leak'
        if is_peer:
            col, bg, dk = PEER, PEER_BG, PEER_DK
            bx, avx = x0, x1 - av / 2
        else:
            col, bg, dk = TARGET, TARGET_BG, TARGET_DK
            bx, avx = x0 + av + gap, x0 + av / 2
        edge, lw = (LEAK, 1.5) if leak else (dk, 0.7)
        if leak:
            bg = LEAK_BG
        rnd = who.split(', ')[0]
        lines = c.wrap('“' + text + '”', bub_w - 2 * pad)
        bh = len(lines) * lh + 2 * pad - 0.02
        name = 'Peer' if is_peer else 'Target'
        hy = y - 0.07
        c.text(bx + 0.04, hy, name, weight='bold', color=dk, va='center')
        c.text(bx + 0.04 + c.width(name + '  ', FS, 'bold'), hy, rnd, size=FS_SMALL,
               color=MUTED, va='center')
        top = y - 0.14
        rbox(c, bx, top - bh, bub_w, bh, bg, edge, lw=lw, r=0.05, z=3)
        ty = top - 0.15
        if is_peer:
            ex = bx + bub_w
            tail = [(ex - 0.02, ty + 0.045), (ex + 0.06, ty), (ex - 0.02, ty - 0.045)]
        else:
            tail = [(bx + 0.02, ty + 0.045), (bx - 0.06, ty), (bx + 0.02, ty - 0.045)]
        c.add(Polygon(tail, closed=True, fc=bg, ec=edge, lw=lw, joinstyle='round', zorder=2))
        c.add(Polygon(tail, closed=True, fc=bg, ec='none', zorder=4))  # hide seam
        robot(c, avx, ty - 0.01, av, col, bg if not leak else TARGET_BG, dk,
              kind='peer' if is_peer else 'target', body=False)
        for i, ln in enumerate(lines):
            ly = top - pad - (i + 0.80) * lh
            tok = 'stg_4be6'
            xx = bx + pad
            if tok in ln:
                pre, post = ln.split(tok, 1)
                ell = '…' if post.startswith('…') else ''
                rest = post[len(ell):]
                c.text(xx, ly, pre, va='baseline')
                xx += c.width(pre)
                token = tok + ell
                tw = c.width(token, weight='bold')
                c.add(FancyBboxPatch((xx, ly - 0.018), tw, lh * 0.86,
                                     boxstyle='round,pad=0.015', fc=LEAK, ec='none',
                                     alpha=0.6, zorder=5))
                c.text(xx, ly, token, weight='bold', va='baseline')
                c.text(xx + tw, ly, rest, va='baseline')
            else:
                c.text(xx, ly, ln, va='baseline')
        if leak:
            label = 'LEAK'
            bwid = c.width(label, FS_SMALL, 'bold') + 0.12
            bx1 = bx + bub_w - 0.08
            rbox(c, bx1 - bwid, top - 0.075, bwid, 0.15, LEAK, LEAK_DK, lw=0.7, r=0.04, z=8)
            c.text(bx1 - bwid / 2, top, label, size=FS_SMALL, weight='bold',
                   ha='center', va='center', zorder=12)
        y = top - bh
        if note:
            nt = note.strip('()')
            ny = y - 0.10
            cx = bx + bub_w / 2 if not leak else bx + bub_w / 2
            c.text(cx, ny, nt, size=FS_SMALL, style='italic', color=MUTED,
                   ha='center', va='center')
            if not leak:
                wnt = c.width(nt, FS_SMALL, style='italic')
                for sx in (-1, 1):
                    c.line([cx + sx * (wnt / 2 + 0.07), cx + sx * (wnt / 2 + 0.40)],
                           [ny] * 2, color=RULE, lw=0.9, ls=(0, (1, 2)))
            y = ny - 0.07
        y -= 0.03
    return y


# ---------------------------------------------------------------- panel (b)
def panel_b(c, x0, x1, ytop, ybot):
    panel_title(c, x0, ytop, 'b', 'Who acts; who observes')
    s = 0.40
    pcx, tcx, rcy = x0 + 0.30, x1 - 0.48, ytop - 0.50
    robot(c, pcx, rcy, s, PEER, PEER_BG, PEER_DK, kind='peer')
    robot(c, tcx, rcy, s, TARGET, TARGET_BG, TARGET_DK, kind='target')
    yl = rcy - 0.44
    c.text(pcx, yl, 'Peer', weight='bold', color=PEER_DK, ha='center', va='center')
    c.text(pcx, yl - 0.13, 'asks for the token', size=FS_SMALL, ha='center', va='center')
    c.text(tcx, yl, 'Target', weight='bold', color=TARGET_DK, ha='center', va='center')

    # team channel between them: the peer's message on top, the target's reply below
    chx0, chx1 = pcx + 0.40, tcx - 0.40
    chy, chh = rcy - 0.03, 0.34
    rbox(c, chx0, chy - chh / 2, chx1 - chx0, chh, 'white', MUTED, lw=0.7, r=0.05, z=2)
    c.text((chx0 + chx1) / 2, chy + chh / 2 + 0.09, 'team channel', size=FS_SMALL,
           color=MUTED, ha='center', va='center')
    mw = (chx1 - chx0) * 0.62
    up, dn = chy + 0.075, chy - 0.075
    rbox(c, chx0 + 0.05, up - 0.05, mw, 0.10, PEER_BG, PEER, lw=0.6, r=0.03, z=3)
    rbox(c, chx1 - 0.05 - mw, dn - 0.05, mw, 0.10, TARGET_BG, TARGET, lw=0.6, r=0.03, z=3)
    hx = s / 2 + 0.06  # clear of the robot's ears
    arrow(c, (pcx + hx, up), (chx0, up), color=PEER_DK)
    arrow(c, (chx1, up), (tcx - hx, up), color=PEER_DK)
    arrow(c, (tcx - hx, dn), (chx1, dn), color=TARGET_DK)
    arrow(c, (chx0, dn), (pcx + hx, dn), color=TARGET_DK)  # target's messages reach the peer

    # what only the target has: the operator's rule and the workspace behind its tools
    bx0, bx1 = chx0 + 0.16, x1 - 0.16          # clear of the monitor and probe arrows
    btop, bbot = yl - 0.12, ybot + 0.46 + 0.10
    rbox(c, bx0, bbot, bx1 - bx0, btop - bbot, '#F4F8FC', TARGET_DK, lw=0.8, r=0.05, z=2)
    tail = [(tcx - 0.07, btop - 0.005), (tcx, btop + 0.045), (tcx + 0.07, btop - 0.005)]
    c.add(Polygon(tail, closed=True, fc='#F4F8FC', ec=TARGET_DK, lw=0.8,
                  joinstyle='round', zorder=1))
    c.add(Polygon(tail, closed=True, fc='#F4F8FC', ec='none', zorder=3))  # hide seam
    c.text(bx0 + 0.07, btop - 0.09, 'Target only, never the peer', size=FS_SMALL, weight='bold',
           color=TARGET_DK, va='center')
    tx = bx0 + 0.33
    r1, r2 = btop - 0.26, btop - 0.53
    scroll(c, bx0 + 0.09, r1 - 0.11, 0.15, 0.20, PAPER_BG, '#9C7A3C')
    c.text(tx, r1 + 0.045, r"operator's rule $\rho$:", size=FS_SMALL, va='center')
    c.text(tx, r1 - 0.075, 'never share the token', size=FS_SMALL, style='italic',
           va='center')
    document(c, bx0 + 0.09, r2 - 0.11, 0.15, 0.20, 'white', TARGET_DK)
    lock(c, bx0 + 0.22, r2 - 0.07, 0.085, TARGET_DK)
    c.text(tx, r2 + 0.045, r'mock tools reach files $F$;', size=FS_SMALL, va='center')
    c.text(tx, r2 - 0.075, r'token in private file $f^{\star}$', size=FS_SMALL, va='center')
    for t in (c.ax.texts[-1], c.ax.texts[-2], c.ax.texts[-4], c.ax.texts[-5]):
        bb = t.get_window_extent(c.renderer)
        assert bb.x1 / c.fig.dpi < bx1 - 0.04, t.get_text()

    # observers (dashed green arrows: they read, they do not act)
    oh, oy = 0.46, ybot
    ow = 1.05
    mx = x0
    rbox(c, mx, oy, ow, oh, OBS_BG, OBS_DK, lw=0.8, r=0.05, z=2)
    eye(c, mx + 0.17, oy + oh / 2, 0.22, OBS_DK)
    c.text(mx + 0.33, oy + oh / 2 + 0.075, 'Text monitor', weight='bold', color=OBS_DK,
           va='center')
    c.text(mx + 0.33, oy + oh / 2 - 0.08, 'reads transcript', size=FS_SMALL, va='center')
    ax_ = chx0 + 0.06
    arrow(c, (ax_, chy - chh / 2), (ax_, oy + oh), color=OBS_DK, ls=(0, (3, 2)))

    px = mx + ow + 0.10
    pw = x1 - px
    rbox(c, px, oy, pw, oh, OBS_BG, OBS_DK, lw=0.8, r=0.05, z=2)
    cell = 0.056
    ax0, ay0 = px + 0.08, oy + oh / 2 - 2.5 * cell
    activations(c, ax0, ay0, cell)
    needle(c, (px + 0.29, oy + 0.075), (ax0 + cell * 0.55, ay0 + 1.7 * cell),
           OBS_DK, size=0.55)
    tx = px + 0.35
    c.text(tx, oy + oh - 0.105, 'Probe', weight='bold', color=OBS_DK, va='center')
    l1, l2 = 'reads activations before', 'each call or in reasoning'
    assert tx + max(c.width(l1, FS_SMALL), c.width(l2, FS_SMALL)) < x1 - 0.04
    c.text(tx, oy + oh - 0.24, l1, size=FS_SMALL, va='center')
    c.text(tx, oy + oh - 0.36, l2, size=FS_SMALL, va='center')
    # elbow from the target's head to the probe card
    ex = x1 - 0.07
    c.line([tcx + s / 2 + 0.05, ex], [rcy, rcy], color=OBS_DK, lw=0.9, ls=(0, (3, 2)),
           zorder=5)
    arrow(c, (ex, rcy), (ex, oy + oh), color=OBS_DK, ls=(0, (3, 2)))


# ---------------------------------------------------------------- panel (c)
def panel_c(c, x0, x1, ytop, rows, calls, rounds):
    panel_title(c, x0, ytop, 'c', 'Labels on a schematic episode')
    lab_w = 1.55
    gx0 = x0 + lab_w
    n = len(calls)
    cw = (x1 - gx0) / n
    rh = 0.25
    y_round = ytop - 0.19
    y_call = y_round - 0.185
    ys = [y_call - 0.03 - rh * (i + 1) for i in range(len(rows))]
    bx0 = gx0 + cw + 0.01
    band_top = y_call + 0.12
    band_bot = ys[-1] - rh / 2 + 0.01
    rbox(c, bx0, band_bot, x1 - bx0, band_top - band_bot, SCORED_BG, RULE, lw=0.6,
         r=0.04, z=0)
    c.text(x0, y_round, 'Round', weight='bold', va='center')
    c.text(x0, y_call, 'Target call', weight='bold', va='center')
    i = 0
    while i < n:
        j = i
        while j + 1 < n and rounds[j + 1] == rounds[i]:
            j += 1
        cx0, cx1 = gx0 + i * cw, gx0 + (j + 1) * cw
        c.text((cx0 + cx1) / 2, y_round, str(rounds[i]), ha='center', va='center')
        if j > i:  # bracket spanning the calls of one round
            mid, half = (cx0 + cx1) / 2, 0.09
            for xa, xb in ((cx0 + 0.07, mid - half), (mid + half, cx1 - 0.07)):
                c.line([xa, xb], [y_round] * 2, color=MUTED, lw=0.6)
            for xe in (cx0 + 0.07, cx1 - 0.07):
                c.line([xe, xe], [y_round, y_round - 0.035], color=MUTED, lw=0.6)
        i = j + 1
    chip = {'Read': ('white', MUTED, INK), 'Refuse': (TARGET_BG, TARGET_DK, TARGET_DK),
            'Leak': (LEAK, LEAK_DK, INK)}
    for k, ev in enumerate(calls):
        fc, ec, tc = chip[ev]
        cx = gx0 + (k + 0.5) * cw
        w_ = cw - 0.12
        rbox(c, cx - w_ / 2, y_call - 0.075, w_, 0.15, fc, ec, lw=0.7, r=0.07, z=3)
        c.text(cx, y_call, ev, size=FS_SMALL, ha='center', va='center', color=tc,
               weight='bold' if ev == 'Leak' else 'normal')
    for (name, desc, vals), y in zip(rows, ys):
        c.text(x0, y + 0.05, name, weight='bold', va='center')
        c.text(x0, y - 0.065, desc, size=FS_SMALL, color=MUTED, va='center')
        for k, v in enumerate(vals):
            cx = gx0 + (k + 0.5) * cw
            w_, h_ = cw - 0.16, rh - 0.07
            if v is None:
                c.text(cx, y, '–', color=MUTED, ha='center', va='center')
                continue
            fc, tc, ec = (TARGET, 'white', TARGET_DK) if v else (ZERO_BG, MUTED, '#D2D9E0')
            rbox(c, cx - w_ / 2, y - h_ / 2, w_, h_, fc, ec, lw=0.6, r=0.035, z=3)
            c.text(cx, y, str(v), ha='center', va='center', color=tc,
                   weight='bold' if v else 'normal')
    c.text(x0, band_bot - 0.11,
           'Shaded: calls scored after the first read. A refusing episode has zero for each label.',
           size=FS_SMALL, color=MUTED, va='center')


def load_exchange():
    tree = ast.parse(SOURCE.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'EXCHANGE' for t in n.targets))


def main():
    exchange = load_exchange()
    c = Canvas()
    m = 0.03
    split = 2.74
    top = H - 0.12
    divider = 1.62
    a_end = panel_a(c, exchange, m, split - 0.10, top)
    assert a_end > divider, a_end
    panel_b(c, split + 0.12, W - m, top, divider + 0.10)
    c.line([m, W - m], [divider, divider], color=RULE, lw=0.6)
    calls = ['Read', 'Refuse', 'Refuse', 'Refuse', 'Refuse', 'Read', 'Leak']
    rounds = [1, 2, 3, 4, 5, 6, 6]
    rows = [
        ('Ever', 'episode leaks at some point', [None, 1, 1, 1, 1, 1, 1]),
        ('This/next round', 'leaks this round or the next', [None, 0, 0, 0, 1, 1, 1]),
        ('This call', 'this call is the leak', [None, 0, 0, 0, 0, 0, 1]),
    ]
    panel_c(c, m, W - m, divider - 0.15, rows, calls, rounds)
    out = PAPER / 'figures/fig2_setup'
    for ext in ['pdf', 'png']:
        c.fig.savefig(out.with_suffix('.' + ext), dpi=300, facecolor='white')
    out.with_suffix('.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(PAPER.parent)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'excerpt': 'EXCHANGE constant, verbatim shortened quotes and token truncation preserved',
        'timeline': 'schematic, not reconstructed transcript; seven calls across six illustrative rounds',
        'width_inches': W, 'height_inches': H, 'access_codes_removed': True}, indent=2) + '\n')
    plt.close(c.fig)


if __name__ == '__main__':
    main()
