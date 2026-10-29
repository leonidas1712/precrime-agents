"""Generate registry-backed figures; pending estimates are never plotted as data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import configure, MODEL_COLORS, DETECTOR_MARKERS, LABEL_SIZE, paper_number

PAPER = Path(__file__).resolve().parents[2]
OUT = PAPER / 'figures'


def registry():
    raw = (PAPER / 'numbers.yaml').read_bytes()
    entries = json.loads(raw)['numbers']
    return {e['macro']: dict(e, key=k) for k, e in entries.items()}, hashlib.sha256(raw).hexdigest()


def get(data, macro):
    return data.get(macro, dict(macro=macro, value=None, status='pending', key='unregistered'))


def draw(ax, x, y, entry, model, kind='activation', horizontal=False, fresh=False, marker=None):
    if entry.get('value') is None:
        return False
    color = MODEL_COLORS[model]
    value = entry['value']
    ci = entry.get('ci')
    error = [[value-ci[0]], [ci[1]-value]] if ci else None
    ax.errorbar(value if horizontal else x, y if horizontal else value,
                xerr=error if horizontal else None, yerr=None if horizontal else error,
                marker=marker or DETECTOR_MARKERS[kind], color=color, markersize=8 if fresh else 6, markeredgewidth=1.5,
                markeredgecolor='#222222' if fresh else color,
                markerfacecolor=color if fresh or entry['status']=='confirmed' else 'white',
                capsize=2, linewidth=1, linestyle='none', zorder=3)
    return True


def value_label(ax, entry, x, y, color, ha='left', va='center', offset=(4, 0)):
    """Write the reported number (as numbers.tex renders it) next to a mark."""
    text = paper_number(entry['macro'], entry['value'])
    ax.annotate(text, (x, y), xytext=offset, textcoords='offset points', ha=ha, va=va,
                fontsize=LABEL_SIZE, color=color, zorder=4)


def whisker_end(entry, side='high'):
    ci = entry.get('ci')
    return (ci[1] if side == 'high' else ci[0]) if ci else entry['value']


def save(fig, stem, rows, sha):
    fig.savefig(OUT/f'{stem}.pdf', bbox_inches=None, facecolor='white')
    fig.savefig(OUT/f'{stem}.png', bbox_inches=None, dpi=180, facecolor='white')
    fields=['panel','label','macro','key','value','ci_low','ci_high','status','dataset','evaluation','run']
    with (OUT/f'{stem}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for panel,label,e in rows:
            ci=e.get('ci') or [None,None]
            w.writerow(dict(panel=panel,label=label,macro=e.get('macro'),key=e['key'],value=e.get('value'),
                            ci_low=ci[0],ci_high=ci[1],**{k:e.get(k) for k in fields[7:]}))
    (OUT/f'{stem}.json').write_text(json.dumps({
        'registry_sha256':sha,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),
        'pending_macros':[e.get('macro') for _,_,e in rows if e.get('value') is None],
        'uncertainty':'source-registry confidence interval, when available; no invented intervals',
        'transformation':'none; values and intervals read directly from numbers.yaml',
        'marker_fill':('not applicable: bars' if stem=='fig3_percall' else ('filled fresh test; hollow development' if stem=='fig1_headline' else 'filled confirmed; hollow exploratory')),
    },indent=2)+'\n')
    plt.close(fig)


def headline(data,sha):
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.25),sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.325,top=.91,wspace=.13)
    rows=[]
    for ax,model,title in zip(axes,['qwen','gpt_oss'],['Qwen3-32B','gpt-oss-20b']):
        prefix='qwen' if model=='qwen' else 'gptoss'
        keys=[prefix+'EverProbe',prefix+'NearProbe',prefix+'CallProbe']
        values=[]
        for x,key in enumerate(keys):
            e=get(data,key);rows.append((title,['Ever','This/next round','This call'][x],e))
            position = x - .18 if x == 2 else x
            if draw(ax,position,None,e,model):
                values.append((position,e['value']))
                if x == 2:value_label(ax,e,position,e['value'],MODEL_COLORS[model],ha='right',offset=(-6,0))
                else:value_label(ax,e,position,whisker_end(e),MODEL_COLORS[model],ha='center',va='bottom',offset=(0,2))
        # Do not bridge an unobserved rung.
        for a,b in zip(values,values[1:]):
            if .8 <= b[0]-a[0] <= 1:ax.plot([a[0],b[0]],[a[1],b[1]],color=MODEL_COLORS[model],lw=.8,alpha=.6)
        fresh=[(2,'qwenFreshCallProbe','activation')] if model=='qwen' else [(2,'gptossFreshCallProbe','activation'),(2,'gptossFreshReasoning','activation')]
        for i,(x,key,kind) in enumerate(fresh):
            e=get(data,key);rows.append((title,'Fresh test: '+key,e))
            fresh_x=x+(0 if model=='gpt_oss' and i==0 else .18)
            draw(ax,fresh_x,None,e,model,kind,fresh=True,
                 marker='D' if key=='gptossFreshReasoning' else 'o')
            if e.get('value') is not None:
                # The caption and legend name these markers; the label sits under the lower whisker.
                if model=='qwen':
                    value_label(ax,e,fresh_x,whisker_end(e,'low'),MODEL_COLORS[model],ha='center',va='top',offset=(0,-2))
                elif key=='gptossFreshReasoning':  # right of the diamond, clear of its whisker
                    value_label(ax,e,fresh_x,e['value'],MODEL_COLORS[model],offset=(8,0))
                else:  # left of the last-prompt marker, below the development point
                    value_label(ax,e,fresh_x,e['value'],MODEL_COLORS[model],ha='right',offset=(-8,-13))
        ax.axhline(.5,color='#777777',ls='--',lw=.8)
        ax.set(xlim=(-.3,2.62),ylim=(0,1.03),xticks=[0,1,2],xticklabels=['Will it\never leak?','This round\nor next?','This call?'],title=title)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('AUROC')
    handles=[
        Line2D([],[],marker='o',mfc='white',color='#333333',ls='none',label='Last-prompt probe: development'),
        Line2D([],[],marker='o',color='#333333',ls='none',label='Last-prompt probe: test episodes'),
        Line2D([],[],marker='D',color=MODEL_COLORS['gpt_oss'],ls='none',label='Reasoning probe: gpt-oss test episodes')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,0),ncol=1,frameon=False,fontsize=9)
    save(fig,'fig1_headline',rows,sha)


def marker_legend(fig, kinds, fresh=False, y=.015):
    names={'activation':'Activation probe', 'text':'Text monitor', 'rule':'Rule', 'control':'Pre-ask control'}
    handles=[Line2D([],[],marker=DETECTOR_MARKERS[k],mfc='#555555' if fresh else 'white',
                    color='#555555',ls='none',label=names[k]) for k in kinds]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),
               ncol=len(kinds),frameon=False,fontsize=9)


def forest(data,sha,stem,title,panels):
    compact=stem=='fig_round_confirmation'
    fig,axes=plt.subplots(1,len(panels),figsize=(3.9,3.0) if compact else (6.5,3.8),squeeze=False)
    fig.subplots_adjust(left=.06,right=.97,bottom=.38 if compact else .31,top=.77,wspace=.40)
    rows=[]
    for ax,(model,panel,items) in zip(axes[0],panels):
        ax.set_title(panel,fontsize=10)
        for y,(label,key,kind) in enumerate(items):
            e=get(data,key);rows.append((panel,label,e))
            if draw(ax,None,y,e,model,kind,horizontal=True):
                value_label(ax,e,whisker_end(e),y,MODEL_COLORS[model])
            ax.text(.02,y-.30,label,va='center',fontsize=10 if compact else 9,
                    bbox=dict(facecolor='white',edgecolor='none',pad=1.5),
                    transform=ax.get_yaxis_transform())
        ax.set(xlim=(0,1.03),ylim=(len(items)-.4,-.8),yticks=[],xlabel='AUROC',xticks=[0,.5,1])
        ax.axvline(.5,color='#777777',ls='--',lw=.8)
        if not compact:
            ax.text(.5,.02,'chance',ha='center',va='bottom',fontsize=9,
                    bbox=dict(facecolor='white',edgecolor='none',pad=1),transform=ax.get_xaxis_transform())
        ax.spines['left'].set_visible(False)
    fig.suptitle(title,fontsize=11,y=.98)
    kinds=list(dict.fromkeys(kind for _,_,items in panels for _,_,kind in items))
    marker_legend(fig,kinds,fresh=compact)
    fig.text(.5,.13,'Whiskers: 95% CI; dashed line: chance',ha='center',fontsize=9)
    save(fig,stem,rows,sha)


def shortcuts(data,sha):
    fig,ax=plt.subplots(figsize=(6.5,3.9))
    fig.subplots_adjust(left=.43,right=.97,bottom=.30,top=.92)
    items=[
        ('Last-prompt probe','gptossCallProbe','activation',0),
        ('Reasoning probe','gptossCallReasoning','activation',1),
        ('Call reads the protected file','gptossReadStep','rule',2),
        ('Keywords in reasoning','gptossKeyword','rule',3),
        ('Before the peer asks\n(other calls)','gptossPlacebo','control',4),
        ('Probe reading only incoming\nmessages and tool results','gptossIncomingProbe','activation',6)]
    rows=[]
    reference=get(data,'gptossCallProbe').get('value')
    for label,key,kind,y in items:
        e=get(data,key);rows.append(('gpt-oss-20b',label,e))
        if draw(ax,None,y,e,'gpt_oss',kind,horizontal=True):
            # Labels go right of the upper whisker unless that would cross the dashed reference line.
            if whisker_end(e) < reference < whisker_end(e) + .1:
                value_label(ax,e,whisker_end(e,'low'),y,MODEL_COLORS['gpt_oss'],ha='right',offset=(-4,0))
            else:value_label(ax,e,whisker_end(e),y,MODEL_COLORS['gpt_oss'])
    ax.spines['bottom'].set_bounds(0,1)
    ax.set(xlim=(0,1.1),ylim=(6.7,-.7),yticks=[i[3] for i in items],
           yticklabels=[i[0] for i in items],xlabel='AUROC',xticks=[0,.5,1])
    ax.tick_params(axis='y',length=0,labelsize=9)
    ax.axvline(.5,color='#777777',ls='-',lw=.8)
    if reference is not None:ax.axvline(reference,color=MODEL_COLORS['gpt_oss'],ls=(0,(6,3)),lw=1)
    ax.axhline(5,color='#bbbbbb',lw=.6,xmax=1/1.1)
    ax.spines['left'].set_visible(False)
    ax.set_title('gpt-oss-20b',fontsize=10)
    lines=[Line2D([],[],color='#777777',lw=.8,label='Chance'),
           Line2D([],[],color=MODEL_COLORS['gpt_oss'],ls=(0,(6,3)),label='Last-prompt probe')]
    fig.legend(handles=lines,loc='lower center',bbox_to_anchor=(.5,.12),ncol=2,frameon=False,fontsize=9)
    marker_legend(fig,['activation','rule','control'],y=.045)
    fig.text(.5,.015,'Whiskers: 95% CI',ha='center',fontsize=9)
    save(fig,'fig4_shortcuts',rows,sha)


def percall(data, sha):
    panels = [
        ('qwen', 'Qwen3-32B', [
            ('Last-prompt probe', 'qwenCallProbe'),
            ('Trained text: latest message', 'qwenTextLatest'),
            ('Zero-shot Jev', 'qwenCallJev')]),
        ('gpt_oss', 'gpt-oss-20b', [
            ('Last-prompt probe', 'gptossCallProbe'),
            ('Reasoning probe', 'gptossCallReasoning'),
            ('Trained text: full input', 'gptossTextFull'),
            ('Zero-shot Jev (reads earlier reasoning)', 'gptossCallJev')])]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.7), sharex=True)
    fig.subplots_adjust(left=.07, right=.97, bottom=.21, top=.92, wspace=.22)
    rows = []
    for ax, (model, title, items) in zip(axes, panels):
        ax.set_title(title, fontsize=10)
        for y, (label, key) in enumerate(items):
            e = get(data, key)
            assert e.get('value') is not None, key
            rows.append((title, label, e))
            value, ci = e['value'], e.get('ci')
            ax.barh(y, value, height=.30, color=MODEL_COLORS[model], alpha=.55)
            if ci:
                ax.errorbar(value, y, xerr=[[value-ci[0]], [ci[1]-value]],
                            fmt='none', ecolor='#222222', capsize=3, linewidth=1)
            value_label(ax, e, whisker_end(e), y, '#222222')
            ax.text(0, y-.31, label, fontsize=9, va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
        ax.axvline(.5, color='#777777', ls='--', lw=.8, zorder=0)
        ax.set(xlim=(0,1.13), ylim=(3.5,-.75), yticks=[], xticks=[0,.5,1], xlabel='AUROC')
        ax.spines['bottom'].set_bounds(0,1)
        ax.spines['left'].set_visible(False)
    fig.text(.5,.025,'Development episodes; whiskers: 95% CI; dashed line: chance',ha='center',fontsize=9)
    save(fig, 'fig3_percall', rows, sha)


def main():
    configure();data,sha=registry();OUT.mkdir(exist_ok=True)
    headline(data,sha)
    forest(data,sha,'fig_round_confirmation','Qwen3-32B test episodes', [('qwen','Will it leak this round or next?', [('Probe','qwenFreshNearProbe','activation'),('Jev','qwenFreshNearJev','text')])])
    percall(data,sha)
    shortcuts(data,sha)


if __name__=='__main__':main()
