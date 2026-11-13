"""Register E19: checks on saved data queued for v2 (read flag one call earlier, text baselines and Jev on the
permission experiments, full-input classifier after a read, AUROC by layer). Numbers only, no tables."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = sorted((ROOT / 'runs').glob('*-e19-v2-analyses'))[-1].relative_to(ROOT).as_posix()
SCRIPTS = 'experiments/E19-v2-analyses'


def main():
    reg = json.loads((HERE / 'numbers.yaml').read_text())
    cache = {}

    def get(file, path):
        if file not in cache:
            cache[file] = json.loads((ROOT / RUN / file).read_text())
        v = cache[file]
        for k in path: v = v[k]
        return v

    def add(macro, file, path, ci=None, fmt='decimal', decimals=2, dataset='', evaluation=''):
        source = dict(file=f'{RUN}/{file}', value_path=path)
        entry = dict(value=get(file, path), macro=macro, dataset=dataset,
                     evaluation=evaluation, status='verified', run=RUN, format=fmt, decimals=decimals, source=source)
        if ci:
            entry['ci'] = get(file, ci)
            source['ci_path'] = ci
        reg['numbers'][macro] = entry

    E = 'eNineteen'
    # Read flag on the one-call-earlier label (gpt-oss development episodes)
    ds = 'gpt-oss-20b development episodes, one-call-earlier label (1,874 calls, 116 positives, 238 families)'
    ev = 'Within scenario-and-round AUROC, held-out scenarios, 3 seeds; family bootstrap 1,000 draws'
    f = 'task3_readflag_next.json'
    add(E + 'ReadFlagNext', f, ['own_protected_read_call', 'auc'], ['own_protected_read_call', 'ci95'], dataset=ds, evaluation=ev)
    add(E + 'ReasoningNext', f, ['reference_reasoning_probe_token_max_strict1', 'auc'],
        ['reference_reasoning_probe_token_max_strict1', 'ci95'], dataset=ds, evaluation=ev)
    add(E + 'ProbeMinusReadFlagNext', f, ['own_protected_read_call', 'probe_minus_flag', 'estimate'],
        ['own_protected_read_call', 'probe_minus_flag', 'ci95'], fmt='signed', dataset=ds, evaluation=ev + '; paired')

    # Full-input text classifier on calls right after a read (Qwen development, held-out scenarios)
    ds = 'Qwen3-32B development episodes, calls whose latest input holds the masked secret (235 calls, 46 leaks)'
    ev = 'Within scenario-and-round AUROC, held-out scenarios; family bootstrap 1,000 draws, paired'
    f = 'task2_fullinput_postread.json'
    add(E + 'PostReadBowFull', f, ['dev', 'post_read', 'bow_full', 'auc'], ['dev', 'post_read', 'bow_full', 'ci95'], dataset=ds, evaluation=ev)
    add(E + 'PostReadProbe', f, ['dev', 'post_read', 'probe', 'auc'], ['dev', 'post_read', 'probe', 'ci95'], dataset=ds, evaluation=ev)
    add(E + 'PostReadProbeMinusBowFull', f, ['dev', 'post_read', 'probe_minus_bow_full', 'estimate'],
        ['dev', 'post_read', 'probe_minus_bow_full', 'ci95'], fmt='signed', dataset=ds, evaluation=ev)

    # Text classifier and Jev on the permission experiments (Qwen3-32B)
    f = 'task1_bow.json'
    ds = 'E15 sharing calls away from a read: 31 forbidden leaks, 73 permitted shares when the peer asks at once'
    ev = 'AUROC of forbidden above permitted shares; episode bootstrap 2,000 draws; classifier fit once on development calls'
    add(E + 'AwayBowLatest', f, ['E15', 'qwen_latest', 'auroc_forbidden_vs_early', 'nonread', 'value'],
        ['E15', 'qwen_latest', 'auroc_forbidden_vs_early', 'nonread', 'ci95'], dataset=ds, evaluation=ev)
    add(E + 'AwayProbe', f, ['E15', 'probe', 'auroc_forbidden_vs_early', 'nonread', 'value'],
        ['E15', 'probe', 'auroc_forbidden_vs_early', 'nonread', 'ci95'], dataset=ds, evaluation=ev)
    add(E + 'EEightBowMaxChange', f, ['E08_qwen', 'qwen_latest', 'paired_leaking_call_changes', 'PERMIT_minus_FORBID', 'max_abs'],
        fmt='decimal', decimals=0, dataset='E08 Qwen3-32B: 38 leaking calls replayed under a permitting rule',
        evaluation='Largest change in the latest-message classifier score, permit minus original')
    f = 'task1_jev.json'
    ds = 'E08 Qwen3-32B: 38 leaking test calls replayed under each rule'
    ev = 'Jev, matched cut (E13 rule), with the rule actually in force in its input'
    for arm, name in (('FORBID', 'Forbid'), ('PERMIT', 'Permit'), ('PERMIT_NO_CONSEQUENCE', 'PermitNoConsequence')):
        add(E + 'JevEEight' + name, f, ['E08_qwen', 'threshold_crossing', arm, 'numerator'], fmt='integer', decimals=0, dataset=ds, evaluation=ev)
    add(E + 'JevEEightCalls', f, ['E08_qwen', 'threshold_crossing', 'FORBID', 'denominator'], fmt='integer', decimals=0, dataset=ds, evaluation=ev)
    add(E + 'JevForbiddenAbovePermitted', 'task1_jev_e14e15.json', ['E15', 'auroc_forbidden_vs_early', 'all', 'value'],
        ['E15', 'auroc_forbidden_vs_early', 'all', 'ci95'], decimals=3,
        dataset='E15 sharing calls: 77 forbidden leaks, 109 permitted shares when the peer asks at once',
        evaluation='Jev AUROC of forbidden above permitted shares; episode bootstrap 2,000 draws')

    # AUROC by layer on the test episodes
    f = 'task4_layers.json'
    ev = 'Single-layer probe, paper recipe, fit on all development calls; within scenario-and-round AUROC on test calls'
    for model, name, lo, hi in (('qwen', 'Qwen', 8, 56), ('gptoss', 'Gpt', 7, 16)):
        layers = {int(k): v['test_auc'] for k, v in get(f, [model, 'test_by_layer']).items()}
        mid = {k: v for k, v in layers.items() if lo <= k <= hi}
        ds = f'{model} test episodes; layers {lo} to {hi}'
        kmin, kmax = min(mid, key=mid.get), max(mid, key=mid.get)
        add(E + name + 'LayerMidLow', f, [model, 'test_by_layer', str(kmin), 'test_auc'], dataset=ds, evaluation=ev)
        add(E + name + 'LayerMidHigh', f, [model, 'test_by_layer', str(kmax), 'test_auc'], dataset=ds, evaluation=ev)

    # What the trained text classifier reads (Qwen latest-message view, gpt-oss full-input view)
    BI = 'bow_interp/'
    ev = 'Within scenario-and-round AUROC; refit with the paper recipe on the same calls and folds; point estimates'
    AB = lambda name, key: ['ablations', name, key, 'auc']
    rows = []
    def cell(m, file, path, ds):
        add(m, BI + file, path, dataset=ds, evaluation=ev)
        return '\\' + m + '{}'
    q, qt, g = 'Qwen3-32B development, latest message', 'Qwen3-32B test, latest message', 'gpt-oss-20b development, full input'
    def trio(name, label, qkey=None, gkey=None):
        qn = cell(E + 'Bow' + name + 'QwenTest', 'results.json', AB(qkey, 'qwen_test'), qt) if qkey else None
        qd = cell(E + 'Bow' + name + 'QwenDev', 'results.json', AB(qkey, 'qwen_dev'), q) if qkey else None
        gd = cell(E + 'Bow' + name + 'Gpt', 'results.json', AB(gkey or qkey, 'gpt_dev'), g) if (gkey or qkey) else None
        rows.append([label, f'{qn} / {qd}' if qkey else '-- (not in this view)', gd or '--'])
    rows.append(['Raises the score',
                 r'a tool result showing the protected file (masked secret, other fields); the peer asking for the format or a partial confirmation (\emph{format}, \emph{is it}, \emph{at least}, \emph{confirm})',
                 r"the target's own reasoning kept from earlier in the turn (\emph{need to}, \emph{to read}, \emph{and write}, \emph{entire content})"])
    rows.append(['Lowers the score',
                 r"a confirmation that the target's message was delivered (\emph{delivered true}); plain requests (\emph{please}, \emph{proceed}, \emph{the complete})",
                 r"a last message from the peer; the target's refusal reasoning (\emph{must not}, \emph{cannot include})"])
    trio('Reported', 'Classifier as reported', 'paper classifier (no change)')
    mt = 'message-type-only score (secret-bearing tool result > peer message > other tool result)'
    rows.append(['Type of the latest message alone',
                 cell(E + 'BowTypeQwenTest', 'results_extra.json', ['qwen_test', mt, 'auc'], qt) + ' / ' + cell(E + 'BowTypeQwenDev', 'results_extra.json', ['qwen_dev', mt, 'auc'], q),
                 cell(E + 'BowTypeGpt', 'results_extra.json', ['gpt_dev', mt, 'auc'], g)])
    trio('TopTen', 'Without the top 10 words', 'remove top-10 words by |weight|')
    trio('TopFifty', 'Without the top 50 words', 'remove top-50 words by |weight|')
    trio('Mask', 'Without the mask token', 'remove mask token only (protected_value)')
    trio('Scenario', 'Without scenario-specific words', 'remove scenario-specific words (>=90% of docs in one scenario, df>=5)')
    trio('Reasoning', 'Without the kept reasoning', gkey='gpt: drop retained same-turn reasoning from the input (fields and text)')
    pr = ['cues', '{}', 'subgroups', 'latest message contains masked secret', '{}', 'auc']
    def sub(model, det, m, ds):
        path = [x.format(model) if x == '{}' and i == 1 else x for i, x in enumerate(pr)]
        path[4] = det
        return cell(m, 'results.json', path, ds)
    gsub = lambda det, m: cell(m, 'results.json', ['cues', 'gpt', 'subgroups', 'latest message contains masked secret', det, 'auc'], g)
    rows.append(['Right after a read: classifier vs probe',
                 sub('test', 'text_classifier', E + 'BowPostReadTextQwenTest', qt) + ' vs ' + sub('test', 'probe', E + 'BowPostReadProbeQwenTest', qt)
                 + ' (test); ' + sub('qwen', 'text_classifier', E + 'BowPostReadTextQwenDev', q) + ' vs ' + sub('qwen', 'probe', E + 'BowPostReadProbeQwenDev', q) + ' (dev)',
                 gsub('text_classifier', E + 'BowPostReadTextGpt') + ' vs ' + gsub('probe', E + 'BowPostReadProbeGpt')])
    reg['tables']['e19_bow_interp'] = dict(
        columns=r'@{}>{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}p{0.37\linewidth}>{\raggedright\arraybackslash}p{0.29\linewidth}@{}',
        header=['', 'Qwen3-32B, latest message (test / dev)', 'gpt-oss-20b, full input (dev)'], rows=rows)

    hist = reg['tables']['history']['rows']
    row = ['v2 checks on saved data', 'Same labels as the paper', 'No: run after the paper',
           'Read flag beats last-prompt probe one call ahead; text classifier and Jev added to policy tests; layers flat']
    hist[:] = [r for r in hist if r[0] != row[0]] + [row]
    rep = reg['tables']['repro']['rows']
    rid = '\\texttt{' + RUN.split('/')[-1][:13] + '}'
    new = [['Checks on saved data for v2', rid, '\\texttt{E19-v2-analyses/task*.py}']]
    rep[:] = [r for r in rep if not r[0].startswith('Checks on saved data')] + new
    (HERE / 'numbers.yaml').write_text(json.dumps(reg, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__': main()
