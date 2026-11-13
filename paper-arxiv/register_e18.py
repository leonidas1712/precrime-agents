"""Register E18: robustness and coverage checks of the per-call results (metric variants, gpt-oss test Jev and text classifier,
per-scenario gpt-oss test, split-seed spread). Tracking log: experiments/E18-robustness/RESULTS.md."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROB = 'runs/20261106-2314-e18-metric-robustness'
BOW = 'runs/20261106-2314-e18-gptoss-bow-test'
JEV = 'runs/20261106-2315-e18-gptoss-jev-test'
EVAL = ('Saved scores only; within scenario x round AUROC weighted by positives (paper metric), and the same with rounds pooled '
        '(scenario only) or with all calls pooled; 1,000 family-bootstrap draws, seed 20261025')
SETS = {'qwen_dev': 'Qwen development episodes', 'gpt_dev': 'GPT-OSS development episodes',
        'qwen_test': 'Qwen fresh test episodes', 'gpt_test': 'GPT-OSS fresh test episodes'}


def main():
    reg = json.loads((HERE / 'numbers.yaml').read_text())
    cache = {}

    def get(file, path):
        if file not in cache:
            cache[file] = json.loads((ROOT / file).read_text())
        v = cache[file]
        for k in path: v = v[k]
        return v

    def add(macro, file, path, ci=None, fmt='decimal', decimals=2, dataset='', evaluation=EVAL, run=None):
        source = dict(file=file, value_path=path)
        entry = dict(value=get(file, path), macro=macro, dataset=dataset, evaluation=evaluation, status='verified',
                     run=run or str(Path(file).parent), format=fmt, decimals=decimals, source=source)
        if ci:
            entry['ci'] = get(file, ci)
            source['ci_path'] = ci
        reg['numbers'][macro] = entry
        return {'key': macro, 'ci': bool(ci)}

    F = ROB + '/robustness.json'
    E = 'eEighteen'
    # 1. metric variants for every headline per-call number
    rows = []
    spec = [('qwen_dev', 'Qwen3-32B', 'development', [('probe', 'last-prompt probe'), ('text_latest', 'text classifier'), ('jev', 'Jev')]),
            ('gpt_dev', 'gpt-oss-20b', 'development', [('probe', 'last-prompt probe'), ('reasoning', 'reasoning probe'),
                                                     ('text_full', 'text classifier'), ('jev', 'Jev')]),
            ('qwen_test', 'Qwen3-32B', 'test', [('probe', 'last-prompt probe'), ('text_bow', 'text classifier'), ('jev', 'Jev')]),
            ('gpt_test', 'gpt-oss-20b', 'test', [('probe', 'last-prompt probe'), ('reasoning', 'reasoning probe')])]
    tag = {'qwen_dev': 'QwenDev', 'gpt_dev': 'GptDev', 'qwen_test': 'QwenTest', 'gpt_test': 'GptTest'}
    det = {'probe': 'Probe', 'reasoning': 'Reasoning', 'text_latest': 'Text', 'text_full': 'Text', 'text_bow': 'Text', 'jev': 'Jev'}
    for s, model, split, dets in spec:
        base = ['results', s]
        m0 = E + tag[s]
        for key, kind in (('scenario_round', 'Round'), ('pooled', 'Pooled')):
            add(m0 + 'Kept' + kind + 'Calls', F, base + ['coverage', key, 'kept_calls'], fmt='integer', decimals=0, dataset=SETS[s])
            add(m0 + 'Pairs' + kind, F, base + ['coverage', key, 'pairs'], fmt='integer', decimals=0, dataset=SETS[s])
        add(m0 + 'KeptGroups', F, base + ['coverage', 'scenario_round', 'kept_groups'], fmt='integer', decimals=0, dataset=SETS[s])
        add(m0 + 'Groups', F, base + ['coverage', 'scenario_round', 'groups'], fmt='integer', decimals=0, dataset=SETS[s])
        for i, (d, label) in enumerate(dets):
            m = m0 + det[d]
            p = base + ['detectors', d]
            cells = [model if i == 0 else '', split if i == 0 else '', label]
            for key, kind in (('scenario_round', 'Round'), ('scenario', 'Scenario'), ('pooled', 'Pooled')):
                cells.append(add(m + kind, F, p + [key, 'auc'], p + [key, 'ci95'], dataset=SETS[s]))
            lo = add(m + 'DropMin', F, p + ['leave_one_group_out', 'min'], dataset=SETS[s])
            hi = add(m + 'DropMax', F, p + ['leave_one_group_out', 'max'], dataset=SETS[s])
            cells.append(f"\\{lo['key']}{{}}--\\{hi['key']}{{}}")
            rows.append(cells)
    reg['tables']['e18_metrics'] = dict(
        columns=r'@{}lllcccc@{}',
        header=['Model', 'Episodes', 'Detector', 'Scenario and round', 'Scenario only', 'All calls', 'One group out'],
        rows=rows)
    # paired gaps quoted in text
    for s, pair, name in [('qwen_test', 'probe_minus_text_bow', 'QwenTestProbeMinusText'),
                          ('qwen_dev', 'probe_minus_text_latest', 'QwenDevProbeMinusText'),
                          ('gpt_dev', 'probe_minus_jev', 'GptDevProbeMinusJev')]:
        for key, kind in (('scenario_round', 'Round'), ('pooled', 'Pooled')):
            add(E + name + kind, F, ['results', s, 'paired', pair, key, 'estimate'], ['results', s, 'paired', pair, key, 'ci95'],
                fmt='signed', dataset=SETS[s])
    # 2. split seeds (development)
    seed_rows = []
    for s, d, label in [('qwen_dev', 'probe', 'Qwen3-32B, last-prompt probe'), ('gpt_dev', 'probe', 'gpt-oss-20b, last-prompt probe'),
                        ('gpt_dev', 'reasoning', 'gpt-oss-20b, reasoning probe')]:
        cells = [label]
        for i, seed in enumerate((42, 43, 44)):
            cells.append(add(E + tag[s] + det[d] + 'Seed' + 'ABC'[i], F, ['results', s, 'detectors', d, 'scenario_round', 'by_seed', i],
                             dataset=SETS[s]))
        cells.append(f"\\{E + tag[s] + det[d]}Round{{}}")
        seed_rows.append(cells)
    reg['tables']['e18_seeds'] = dict(columns=r'@{}lcccc@{}', header=['Detector (development)', 'Seed 42', 'Seed 43', 'Seed 44', 'Mean'],
                                      rows=seed_rows)
    # 3. gpt-oss test per scenario
    names = [('deploy-token-v2', 'Deploy token'), ('dispatch-integration-code', 'Dispatch integration code'),
             ('handoff-exception-note', 'Handoff exception note'), ('procurement-reserve-price', 'Procurement reserve price'),
             ('service-capacity-figure', 'Service capacity figure'), ('service-rollout-threshold', 'Service rollout threshold')]
    sc_rows = []
    for key, label in names:
        m = E + 'GptTestScen' + ''.join(w.title() for w in key.split('-') if w.isalpha())
        p = ['results', 'gpt_test', 'detectors']
        cells = [label, add(m + 'Calls', F, p + ['probe', 'per_scenario', key, 'calls'], fmt='integer', decimals=0, dataset=SETS['gpt_test']),
                 add(m + 'Leaks', F, p + ['probe', 'per_scenario', key, 'leaks'], fmt='integer', decimals=0, dataset=SETS['gpt_test']),
                 add(m + 'KeptGroups', F, p + ['probe', 'per_scenario', key, 'kept', 'kept_groups'], fmt='integer', decimals=0,
                     dataset=SETS['gpt_test']),
                 add(m + 'Probe', F, p + ['probe', 'per_scenario', key, 'auc'], dataset=SETS['gpt_test']),
                 add(m + 'Reasoning', F, p + ['reasoning', 'per_scenario', key, 'auc'], dataset=SETS['gpt_test'])]
        sc_rows.append(cells)
    reg['tables']['e18_gpt_scenarios'] = dict(columns=r'@{}lrrrcc@{}',
                                              header=['Scenario', 'Calls', 'Leaks', 'Groups kept', 'Last-prompt', 'Reasoning'],
                                              rows=sc_rows)
    # 4. gpt-oss test: Jev and the text classifier (fills Table 1)
    tD = 'GPT-OSS fresh test episodes'
    add('gptossFreshCallBow', BOW + '/metrics.json', ['metrics', 'bow_full_input', 'within_round_auc'],
        ['metrics', 'bow_full_input', 'within_round_ci95'], dataset=tD,
        evaluation='bag-of-words (full input, secret masked) fit once on all 1,995 development calls as in E12, scored once on the 594 test calls/26 positives; within scenario and round; 1000 family-bootstrap draws')
    add('gptossFreshCallBowLatest', BOW + '/metrics.json', ['metrics', 'bow_latest_input', 'within_round_auc'],
        ['metrics', 'bow_latest_input', 'within_round_ci95'], dataset=tD, evaluation='as gptossFreshCallBow, latest-message view')
    add('gptossFreshCallJev', JEV + '/metrics.json', ['metrics', 'jev', 'within_round_auc'], ['metrics', 'jev', 'within_round_ci95'],
        dataset=tD, evaluation='Jev zero-shot, development prompt and input view unchanged, on the same 594 test calls; within scenario and round')
    add('gptossFreshCallJevPooled', JEV + '/metrics.json', ['metrics', 'jev', 'pooled_auc'], ['metrics', 'jev', 'pooled_ci95'],
        dataset=tD, evaluation='as gptossFreshCallJev, all calls pooled')
    for pk, macro in [('last_prompt_minus_bow_full_input', 'gptossFreshCallProbeMinusBow'),
                      ('reasoning_minus_bow_full_input', 'gptossFreshReasoningMinusBow')]:
        add(macro, BOW + '/metrics.json', ['paired', pk, 'within_round'], ['paired', pk, 'within_round_ci95'], fmt='signed',
            dataset=tD, evaluation='paired family bootstrap, within scenario and round, same 594 test calls')
    op = ['operating_point', 'reasoning_probe']
    add('gptossFreshMatchedJevCaught', JEV + '/metrics.json', op + ['jev_caught'], fmt='percent', decimals=0, dataset=tD,
        evaluation='Jev cutoff flagging no more refusing-episode calls than the frozen reasoning-probe threshold (12 of 490)')
    add('gptossFreshMatchedJevCaughtCount', JEV + '/metrics.json', op + ['jev_caught_n'], fmt='integer', decimals=0, dataset=tD)
    add('gptossFreshMatchedProbeRefusing', JEV + '/metrics.json', op + ['probe_refusing'], fmt='percent', decimals=1, dataset=tD)
    # 5. ladder on the test episodes (probes fit on development only, scored once)
    LAD = 'runs/20261107-0013-e18-test-ladder/metrics.json'
    for key, macro in [('qwen/eventual', 'eEighteenQwenTestEver'), ('qwen/round_near', 'eEighteenQwenTestNear'),
                       ('qwen/strict1', 'eEighteenQwenTestStrict'), ('gpt/eventual_any', 'eEighteenGptTestEver'),
                       ('gpt/round_any', 'eEighteenGptTestNear'), ('gpt/strict1', 'eEighteenGptTestStrict')]:
        ds = 'Qwen fresh test episodes' if key.startswith('qwen') else 'GPT-OSS fresh test episodes'
        add(macro, LAD, [key, 'within_round_auc'], [key, 'within_round_ci95'], dataset=ds, run=str(Path(LAD).parent),
            evaluation='last-prompt probe, dev ladder recipe and labels, fit on all development episodes (3 split seeds, z-scored mean), scored once on the test calls; within scenario and round; 1000 family-bootstrap draws')
        add(macro + 'Positives', LAD, [key, 'positives'], fmt='integer', decimals=0, dataset=ds, run=str(Path(LAD).parent))
    add('eEighteenQwenFreshRoundChannelCheck', LAD, ['qwen/round_channel_eligible', 'within_round_auc'], dataset='Qwen fresh test episodes',
        run=str(Path(LAD).parent), evaluation='original channel round label on the original 1,787-call cohort; matches qwenRoundFreshProbe')
    # 6. regularisation sweep (development, held-out scenarios, refit with the same recipe except C)
    CLP = 'runs/20261107-0013-e18-c-sweep-lastprompt/metrics.json'
    CRS = sorted((ROOT / 'runs').glob('*-e18-c-sweep-reasoning/metrics.json'))
    CRS = str(CRS[-1].relative_to(ROOT)) if CRS else None
    words = {'0.01': 'Hundredth', '0.03': 'ThreeHundredths', '0.1': 'Tenth', '0.3': 'ThreeTenths', '1': 'One', '10': 'Ten'}
    c_rows = []
    for c, w in words.items():
        cells = [c + (' (recipe)' if c == '0.1' else '')]
        for file, key, m in [(CLP, 'qwen_lastprompt', 'QwenProbe'), (CLP, 'gpt_lastprompt', 'GptProbe'), (CRS, 'gpt_reasoning', 'GptReasoning')]:
            if file is None:
                cells.append('--'); continue
            ds = 'Qwen development episodes' if key.startswith('qwen') else 'GPT-OSS development episodes'
            cells.append(add('eEighteenC' + w + m, file, [key + '/C=' + c, 'within_round_auc'], [key + '/C=' + c, 'within_round_ci95'],
                             dataset=ds, run=str(Path(file).parent),
                             evaluation='held-out scenarios, recipe unchanged except the L2 strength C; layer reselected per fold; mean of 3 split seeds'))
        c_rows.append(cells)
    reg['tables']['e18_csweep'] = dict(columns=r'@{}lccc@{}', header=['C', 'Qwen3-32B last-prompt', 'gpt-oss-20b last-prompt', 'gpt-oss-20b reasoning'],
                                       rows=c_rows)
    # history and reproducibility index
    hist = reg['tables']['history']['rows']
    row = ['Robustness and coverage checks', 'Same labels; three AUROC variants; gpt-oss test Jev and text classifier',
           'No: chosen after the results, to answer reviewer questions', 'No conclusion changes; Jev below chance only within round']
    hist[:] = [r for r in hist if r[0] != row[0]] + [row]
    rep = reg['tables']['repro']['rows']
    new = [['Robustness: metric variants and seeds', '\\texttt{20261106-2314}', '\\texttt{E18-robustness/metric\\_robustness.py}'],
           ['Robustness: ladder on test episodes', '\\texttt{20261107-0013}', '\\texttt{E18-robustness/test\\_ladder.py}'],
           ['Robustness: regularisation sweep', '\\texttt{20261107-0013}', '\\texttt{E18-robustness/c\\_sweep.py}'],
           ['Robustness: gpt-oss test text and Jev', '\\texttt{20261106-2314/2315}', '\\texttt{E18-robustness/gptoss\\_test\\_text\\_jev.py}']]
    rep[:] = [r for r in rep if not r[0].startswith('Robustness:')] + new
    (HERE / 'numbers.yaml').write_text(json.dumps(reg, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__': main()
