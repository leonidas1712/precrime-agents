"""Check every headline number quoted in the v2 report against the JSON outputs (rounded as quoted)."""
import json
from pathlib import Path
D = Path(__file__).resolve().parent
J = {n: json.loads((D / f'{n}.json').read_text()) for n in ['task1_bow', 'task1_extra', 'task2_fullinput_postread', 'task3_readflag_next', 'task4_layers', 'task5_power', 'task1_jev_estimate']}
fails = []


def get(n, *path):
    v = J[n]
    for p in path: v = v[p]
    return v


def eq(label, value, expected, nd=2):
    ok = (round(value, nd) == expected) if isinstance(expected, float) else (value == expected)
    if not ok: fails.append((label, value, expected))


b = 'task1_bow'
# reference
eq('qwen latest test auc', get(b, 'qwen_test_reference', 'auroc_within_round', 'qwen_latest'), 0.957, 3)
eq('qwen full test auc', get(b, 'qwen_test_reference', 'auroc_within_round', 'qwen_full'), 0.78)
eq('repro', get(b, 'qwen_test_reference', 'reproduction_max_abs_diff_vs_saved_E12_bow'), 0.0)
eq('latest leaks flagged', get(b, 'qwen_test_reference', 'flag_rates', 'qwen_latest', 'leaking_calls'), '25/38')
eq('full leaks flagged', get(b, 'qwen_test_reference', 'flag_rates', 'qwen_full', 'leaking_calls'), '7/38')
eq('probe refusing', get(b, 'qwen_test_reference', 'probe_refusing_flag_rate'), 0.055, 3)
for k, v in [('probe', 0.88), ('qwen_latest', 0.91), ('qwen_full', 0.21)]:
    eq('sd ' + k, get(b, 'qwen_test_reference', 'score_sd_test_calls', k), v)
# E08 qwen
E = get(b, 'E08_qwen')
eq('e08 probe permit', E['probe']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['median'], 0.00)
eq('e08 probe permit ci', [round(x, 2) for x in E['probe']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['ci95']], [-0.03, 0.07])
eq('e08 probe neutral', E['probe']['paired_leaking_call_changes']['NEUTRAL_EDIT_minus_FORBID']['median'], -0.02)
eq('e08 probe nocons', E['probe_with_no_consequence']['paired_leaking_call_changes']['PERMIT_NO_CONSEQUENCE_minus_FORBID']['median'], 0.12)
eq('e08 latest zero', E['qwen_latest']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['share_exactly_zero'], 1.0)
eq('e08 latest nocons zero', E['qwen_latest']['paired_leaking_call_changes']['PERMIT_NO_CONSEQUENCE_minus_FORBID']['share_exactly_zero'], 1.0)
eq('e08 full permit', E['qwen_full']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['median'], 0.003, 3)
eq('e08 full permit ci', [round(x, 3) for x in E['qwen_full']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['ci95']], [0.002, 0.004])
eq('e08 full nocons', E['qwen_full']['paired_leaking_call_changes']['PERMIT_NO_CONSEQUENCE_minus_FORBID']['median'], 0.019, 3)
eq('e08 full max', E['qwen_full']['paired_leaking_call_changes']['PERMIT_NO_CONSEQUENCE_minus_FORBID']['max_abs'], 0.05)
for det, vals in [('probe', [0.98, 0.97, 0.98]), ('qwen_latest', [0.96, 0.96, 0.96]), ('qwen_full', [0.81, 0.81, 0.81])]:
    eq('e08 auc ' + det, [round(E[det]['auc'][c]['within_stratum']['value'], 2) for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT')], vals)
eq('e08 probe nocons auc', E['probe_with_no_consequence']['auc']['PERMIT_NO_CONSEQUENCE']['within_stratum']['value'], 0.99)
eq('e08 cross probe', [E['probe_with_no_consequence']['threshold_crossing'][c]['numerator'] for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE')], [31, 32, 31, 34])
eq('e08 cross latest', [E['qwen_latest']['threshold_crossing'][c]['numerator'] for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE')], [25, 25, 25, 25])
eq('e08 cross full', [E['qwen_full']['threshold_crossing'][c]['numerator'] for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE')], [7, 7, 7, 7])
eq('e08 calls', (E['probe']['calls']['leaking'], E['probe']['calls']['non_leaking'], E['probe']['calls']['families_leaking']), (38, 112, 38))
# E08 gpt
G = get(b, 'E08_gptoss')
eq('g probe permit', G['probe']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['median'], 0.00)
eq('g bow permit', G['gpt_full']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['median'], 0.002, 3)
eq('g bow max', G['gpt_full']['paired_leaking_call_changes']['PERMIT_minus_FORBID']['max_abs'], 0.005, 3)
eq('g auc probe', [round(G['probe']['auc'][c]['within_stratum']['value'], 2) for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT')], [0.81, 0.82, 0.83])
eq('g auc bow', [round(G['gpt_full']['auc'][c]['within_stratum']['value'], 2) for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT')], [0.84, 0.84, 0.84])
eq('g cross probe', [G['probe']['threshold_crossing'][c]['numerator'] for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT')], [8, 8, 6])
eq('g cross bow', [G['gpt_full']['threshold_crossing'][c]['numerator'] for c in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT')], [3, 3, 3])
eq('g test bow auc', G['threshold_matching']['bow_test_auroc_within_round'], 0.84)
eq('g test probe auc', G['threshold_matching']['probe_test_auroc_within_round'], 0.82)
eq('g refusing', (G['threshold_matching']['refusing_episode_calls'], G['threshold_matching']['probe_refusing_flag_rate']), (490, 0.0))
eq('g sd', (round(G['threshold_matching']['score_sd_test_calls']['probe'], 2), round(G['threshold_matching']['score_sd_test_calls']['gpt_full'], 2)), (0.81, 0.13))
# E14
F = get(b, 'E14')
for det, row in [('probe', (9, 0.97, 12, 0.92, 8, 0.42, -1.36)), ('qwen_latest', (8, 0.94, 7, 0.96, 4, 0.47, -1.39)), ('qwen_full', (0, 0.65, 1, 0.76, 0, 0.58, -0.11))]:
    x = F[det]
    got = (x['FORBID_dispatch_test']['leaking_calls']['over_threshold'], round(x['FORBID_dispatch_test']['within_round_auroc'], 2),
           x['PERMIT']['sharing_calls']['over_threshold'], round(x['PERMIT']['within_round_auroc'], 2),
           x['PERMIT_NO_CONSEQUENCE']['sharing_calls']['over_threshold'], round(x['PERMIT_NO_CONSEQUENCE']['within_round_auroc'], 2),
           round(x['PERMIT_NO_CONSEQUENCE']['median_diff_vs_forbidden_leaks'], 2))
    eq('e14 ' + det, got, row)
# E15
S = get(b, 'E15')
for det, row in [('probe', (45, 35, 91, 19, 24, 1, -28)), ('qwen_latest', (46, 36, 91, 4, 2, 0, -10)), ('qwen_full', (11, 6, 18, 7, 9, 1, -10))]:
    x = S[det]
    got = (x['follows_read']['FORBID']['over'], x['follows_read']['EARLY']['over'], x['follows_read']['LATE']['over'],
           x['nonread']['FORBID']['over'], x['nonread']['EARLY']['over'], x['nonread']['LATE']['over'], round(x['diff_EARLY_minus_FORBID_nonread']['points']))
    eq('e15 ' + det, got, row)
    eq('e15 n ' + det, (x['nonread']['FORBID']['n'], x['nonread']['EARLY']['n'], x['nonread']['LATE']['n'], x['all_sharing']['FORBID']['n'], x['all_sharing']['EARLY']['n'], x['all_sharing']['LATE']['n']), (31, 73, 7, 77, 109, 98))
for det, (a, r, al) in [('probe', (0.70, 0.09, 0.61)), ('qwen_latest', (0.84, 0.60, 0.75)), ('qwen_full', (0.71, 0.56, 0.64))]:
    x = S[det]['auroc_forbidden_vs_early']
    eq('e15 auc ' + det, (round(x['nonread']['value'], 2), round(x['follows_read']['value'], 2), round(x['all']['value'], 2)), (a, r, al))
eq('e15 hedge latest', round(S['qwen_latest_nohedge']['auroc_forbidden_vs_early']['nonread']['value'], 3), 0.843, 3)
eq('e15 latest', round(S['qwen_latest']['auroc_forbidden_vs_early']['nonread']['value'], 3), 0.842, 3)
eq('e15 full', round(S['qwen_full']['auroc_forbidden_vs_early']['nonread']['value'], 3), 0.710)
eq('e15 full nohedge', round(S['qwen_full_nohedge']['auroc_forbidden_vs_early']['nonread']['value'], 3), 0.729, 3)
eq('e15 full noasst', round(S['qwen_full_noasst']['auroc_forbidden_vs_early']['nonread']['value'], 3), 0.727, 3)
eq('hedge feats', (get(b, 'hedge_vocabulary', 'qwen_latest', 'n_features'), get(b, 'hedge_vocabulary', 'qwen_full', 'n_features')), (206, 741))
inp = get(b, 'E15_sharing_call_inputs')
eq('hedge earlier', (inp['FORBID']['hedge_regex_in_earlier_assistant_text'], inp['EARLY']['hedge_regex_in_earlier_assistant_text'], inp['LATE']['hedge_regex_in_earlier_assistant_text']), (51, 0, 0))
X = J['task1_extra']
eq('extra', (X['FORBID']['peer message'], X['EARLY']['peer message'], X['EARLY']['other tool result'], X['FORBID']['round>=3'], X['EARLY']['round>=3'], X['FORBID']['peer asks for format / partial / confirm'], X['EARLY']['peer asks for format / partial / confirm']), (29, 55, 18, 22, 0, 13, 0))
# Task 2
T = J['task2_fullinput_postread']
for sp, g, row in [('dev', 'post_read', (235, 46, 0.85, 0.57, 0.5, 0.29)), ('test', 'post_read', (128, 21, 0.71, 0.64, 0.5, 0.07)),
                   ('dev', 'all_calls', (2976, 66, 0.96, 0.71, 0.93, 0.25)), ('test', 'all_calls', (1825, 38, 0.98, 0.78, 0.96, 0.20)),
                   ('dev', 'not_post_read', (2741, 20, 0.91, 0.67, 0.82, 0.23)), ('test', 'not_post_read', (1697, 17, 0.95, 0.71, 0.92, 0.24))]:
    x = T[sp][g]
    eq(f't2 {sp} {g}', (x['calls'], x['leaking'], round(x['probe']['auc'], 2), round(x['bow_full']['auc'], 2), round(x['bow_latest']['auc'], 2), round(x['probe_minus_bow_full']['estimate'], 2)), row)
eq('t2 dev ci', [round(v, 2) for v in T['dev']['post_read']['probe_minus_bow_full']['ci95']], [-0.01, 0.66])
eq('t2 cells', (T['dev']['post_read_tie_check_full']['cells_with_leaking_and_nonleaking'], T['dev']['post_read_tie_check_full']['leaking_calls_in_those_cells'], T['dev']['post_read_tie_check_full']['nonleaking_calls_in_those_cells'], T['dev']['post_read_tie_check_full']['mixed_cells_where_bow_scores_differ'],
                T['test']['post_read_tie_check_full']['cells_with_leaking_and_nonleaking'], T['test']['post_read_tie_check_full']['leaking_calls_in_those_cells'], T['test']['post_read_tie_check_full']['nonleaking_calls_in_those_cells'], T['test']['post_read_tie_check_full']['mixed_cells_where_bow_scores_differ']), (15, 33, 24, 15, 5, 7, 6, 5))
eq('t2 paper', T['dev']['paper_check_all_calls_full_input']['recomputed'] == T['dev']['paper_check_all_calls_full_input']['saved_run'], True)
# Task 3
R = J['task3_readflag_next']
eq('t3 n', (R['rows'], R['positives'], R['families']), (1874, 116, 238))
eq('t3 probe', round(R['probe']['auc'], 2), 0.62)
eq('t3 flag', (round(R['own_protected_read_call']['auc'], 2), [round(v, 2) for v in R['own_protected_read_call']['ci95']]), (0.85, [0.82, 0.89]))
eq('t3 diff', (round(R['own_protected_read_call']['probe_minus_flag']['estimate'], 2), [round(v, 2) for v in R['own_protected_read_call']['probe_minus_flag']['ci95']]), (-0.23, [-0.30, -0.17]))
eq('t3 counts', (R['own_protected_read_call']['positives_flagged'], R['own_protected_read_call']['negatives_flagged']), (108, 49))
eq('t3 input flags', (round(R['latest_input_is_protected_read']['auc'], 2), round(R['previous_call_read_protected_file']['auc'], 2)), (0.40, 0.40))
eq('t3 reasoning', round(R['reference_reasoning_probe_token_max_strict1']['auc'], 2), 0.88)
eq('t3 paper', R['paper_check']['recomputed'] == R['paper_check']['saved'], True)
# Task 4
L = J['task4_layers']
q = L['qwen']['test_by_layer']; g = L['gptoss']['test_by_layer']
eq('t4 q16 48', (round(q['16']['test_auc'], 2), round(q['48']['test_auc'], 2)), (0.95, 0.97))
eq('t4 g6 13', (round(g['6']['test_auc'], 2), round(g['13']['test_auc'], 2)), (0.78, 0.85))
eq('t4 q range mid', (round(min(q[str(l)]['test_auc'] for l in range(8, 57, 4)), 2), round(max(q[str(l)]['test_auc'] for l in range(8, 57, 4)), 2)), (0.95, 0.97))
eq('t4 q range all', (round(min(v['test_auc'] for v in q.values()), 2), round(max(v['test_auc'] for v in q.values()), 2)), (0.91, 0.97))
eq('t4 g range all', (round(min(v['test_auc'] for v in g.values()), 2), round(max(v['test_auc'] for v in g.values()), 2)), (0.72, 0.85))
mid = ['7', '9', '10', '12', '13', '15', '16']
eq('t4 g range mid', (round(min(g[l]['test_auc'] for l in mid), 2), round(max(g[l]['test_auc'] for l in mid), 2)), (0.80, 0.85))
eq('t4 val', (round(L['qwen']['validation_by_seed']['43']['16'], 3), round(L['qwen']['validation_by_seed']['43']['48'], 3), round(L['qwen']['validation_by_seed']['42']['48'], 2)), (0.996, 0.896, 1.0))
eq('t4 ref', (round(L['reference']['qwen_frozen_ensemble_test'], 2), round(L['reference']['gptoss_frozen_ensemble_test'], 2)), (0.98, 0.82))
mx = max(max(v.get('frozen_weight_check_max_abs_coef_diff', {0: 0}).values()) for v in list(q.values()) + list(g.values()))
eq('t4 weight check <= 0.0012', mx <= 0.0012, True)
# Task 5
C = {(c['auc'], c['n_pos'], c['ratio'], c['icc']): c for c in J['task5_power']['cells']}
eq('t5 0.65 1:1 120', round(C[(0.65, 120, 1, 0.3)]['power_exclude_half'], 2), 0.83)
eq('t5 0.65 2:1 80', round(C[(0.65, 80, 2, 0.3)]['power_exclude_half'], 2), 0.78)
eq('t5 0.65 2:1 100', round(C[(0.65, 100, 2, 0.3)]['power_exclude_half'], 2), 0.91)
eq('t5 0.65 1:1 40 width', round(C[(0.65, 40, 1, 0.3)]['mean_ci_width'], 2), 0.44)
eq('t5 icc0', [round(C[(0.65, n, 1, 0.0)]['power_exclude_half'], 2) for n in (60, 100, 150)], [0.51, 0.82, 0.98])
eq('t5 icc6', [round(C[(0.65, n, 1, 0.6)]['power_exclude_half'], 2) for n in (60, 100, 150)], [0.33, 0.65, 0.89])
eq('t5 cov', (round(min(c['coverage_of_true'] for c in C.values()), 2), round(max(c['coverage_of_true'] for c in C.values()), 2)), (0.88, 0.99))
# Jev estimate
je = J['task1_jev_estimate']
eq('jev est', (je['unique_payloads'], round(je['worst_case_usd'], 2), round(je['expected_usd_at_observed_mean_cost'], 2)), (3220, 0.66, 0.42))
eq('jev spend file absent', not (D / 'jev_spend.json').exists(), True)

print('FAILURES:', len(fails))
for f in fails: print(f)
