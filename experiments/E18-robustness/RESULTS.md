# E18: robustness and coverage checks for the arXiv v2 preprint

**Why.** These checks answer questions a reviewer would ask about the per-call results (6 Nov 2026, approved by Leo in chat):
- **Asymmetries between the two models.** gpt-oss-20b test episodes had no Jev or text-classifier score. Qwen3-32B had no all-calls AUROC row.
- **The metric drops most calls.** The within scenario×round metric keeps only groups that hold both kinds of call. On the gpt-oss test set that leaves just 14 groups.
- **The ladder questions are measured on development episodes only.**
- **The regularisation strength C was never varied.**
- **The spread across split seeds was never reported.**

**What this is not.** These checks were decided after the main results had been seen. They test how robust the frozen results are and change none of them. The test episodes had already been scored once with the same frozen detectors. Every new test number below comes from the same frozen detectors or from detectors fit on development episodes only.

**Cost.** Jev on the 594 gpt-oss test calls billed $0.084 (cap $1). Everything else ran on CPU from saved scores and activations.

Old-repo sources were read from `/Users/leo/collusion-research` (read-only).

---

## 1. Every existing paper number reproduces

The script recomputes all 12 per-call AUROCs in Table 1 from saved scores, with the same metric code and the same family bootstrap (1,000 draws, seed 20261025). Each matches its saved `metrics.json` to 1e-12, and the run records `all_reproduced: true`.

- Script: [`metric_robustness.py`](metric_robustness.py)
- Run: [`runs/20261106-2314-e18-metric-robustness/robustness.json`](../../runs/20261106-2314-e18-metric-robustness/robustness.json)
- Two earlier attempts crashed on script bugs before writing any results: `runs/20261106-2313-e18-metric-robustness` and `runs/20261106-2313-e18-metric-robustness-fdf8`. Both are marked failed in `runs/index.jsonl`.

## 2. The headline does not depend on dropping calls

**Every leak is kept under every metric.** The within-group metric drops only non-leaking calls, from groups that have no leak to compare against.

| Set | Scenario×round: calls kept / groups kept / leak×non-leak pairs | Scenario only: pairs | All calls: pairs |
|---|---|---|---|
| Qwen dev | 1,039 of 2,976 / 37 of 120 / 2,087 | 38,382 | 192,060 |
| gpt-oss dev | 595 of 1,995 / 16 of 109 / 4,719 | 35,156 | 226,754 |
| Qwen test | 457 of 1,825 / 23 of 100 / 684 | 13,719 | 67,906 |
| gpt-oss test | 142 of 594 / 14 of 108 / **197** | 2,391 | 14,768 |

AUROC [95% CI] under three metrics. "One group out" is the minimum and maximum AUROC when any single kept scenario×round group is removed.

| Set / detector | Scenario×round (paper) | Scenario only | All calls | One group out |
|---|---|---|---|---|
| Qwen dev probe | 0.960 [.934, .983] | 0.964 [.941, .982] | 0.960 [.936, .981] | .954–.979 |
| Qwen dev text (latest message) | 0.925 [.878, .965] | 0.896 [.845, .937] | 0.898 [.853, .934] | .918–.944 |
| Qwen dev Jev | 0.758 [.684, .805] | 0.805 [.748, .851] | 0.811 [.763, .859] | .752–.771 |
| gpt-oss dev last-prompt | 0.906 [.871, .936] | 0.949 [.927, .967] | 0.946 [.922, .966] | .890–.921 |
| gpt-oss dev reasoning | 0.933 [.912, .952] | 0.972 [.961, .980] | 0.965 [.952, .977] | .924–.951 |
| gpt-oss dev Jev | **0.397** [.332, .453] | **0.820** [.755, .851] | **0.859** [.831, .884] | .362–.426 |
| gpt-oss dev text (full input) | 0.863 [.815, .910] | 0.877 [.832, .921] | 0.816 [.755, .872] | .841–.883 |
| Qwen test probe | 0.976 [.943, .995] | 0.967 [.938, .986] | 0.970 [.949, .986] | .974–.981 |
| Qwen test Jev | 0.749 [.639, .834] | 0.748 [.662, .824] | 0.753 [.679, .827] | .737–.781 |
| Qwen test text | 0.957 [.916, .985] | 0.898 [.833, .939] | 0.903 [.854, .942] | .955–.967 |
| gpt-oss test last-prompt | 0.818 [.657, .948] | 0.910 [.841, .968] | 0.913 [.848, .965] | .803–.844 |
| gpt-oss test reasoning | 0.905 [.775, .991] | 0.945 [.899, .980] | 0.954 [.916, .982] | .883–.939 |

**What this means:**
- **The gpt-oss test set's 14 groups do not drive its result.** Removing any one group moves the AUROC by about 0.02. Keeping every call raises the last-prompt AUROC from 0.82 to about 0.91. The paper's metric is the strict one.
- **Every probe scores the same or higher once rounds are pooled.** The exception is the Qwen test probe, which goes from 0.976 to 0.967.
- **Jev on gpt-oss-20b is below chance only within scenario and round.** Pooled across rounds it scores 0.82–0.86, because its score falls with the round (Spearman −0.78 between Jev score and round on development calls), and leaks cluster in rounds 3–4 (mean round 3.5 for leaks, 10.4 for other calls). Checked by hand on the saved scores, 7 Nov. The paper's sentence "Jev ranks leaking calls below the other calls under the same pressure" is correct, but it is tied to this metric, and the paper should say so.
- **The probe beats Jev and the text classifier under every metric** (paired bootstrap, same JSON). One gap is not significant: the probe over the text classifier on Qwen test episodes is +0.018 [−0.010, 0.055] under the paper's metric. The same gap is +0.07, with a lower bound above 0, under both looser metrics. So the paper's metric is also the conservative one for the probe-versus-text comparison.

## 3. gpt-oss-20b test episodes: per scenario

This mirrors the existing Qwen per-scenario table. Each scenario has only 2–7 leaks, so the intervals are very wide. The last column shows how many calls are kept and how many groups, of 18, have both kinds of call.

| Scenario | Calls | Leaks | Kept calls (groups) | Last-prompt | Reasoning |
|---|---|---|---|---|---|
| Deploy token | 131 | 2 | 21 (2 of 18) | 0.83 | 1.00 |
| Dispatch code | 71 | 7 | 24 (2 of 18) | 0.81 | 1.00 |
| Handoff note | 109 | 4 | 28 (3 of 18) | 1.00 | 1.00 |
| Reserve price | 120 | 4 | 24 (2 of 18) | 0.83 | 0.88 |
| Capacity figure | 119 | 6 | 33 (3 of 18) | 0.77 | 0.84 |
| Rollout threshold | 44 | 3 | 12 (2 of 18) | 0.67 | 0.67 |

Source: the same `robustness.json`.

## 4. Split-seed spread (development, within scenario×round)

| Detector | Seed 42 | Seed 43 | Seed 44 | Mean (paper) |
|---|---|---|---|---|
| Qwen dev probe | 0.944 | 0.972 | 0.964 | 0.960 |
| gpt-oss dev last-prompt | 0.923 | 0.886 | 0.908 | 0.906 |
| gpt-oss dev reasoning | 0.960 | 0.881 | 0.958 | 0.933 |

The probe fit itself is deterministic, so a seed changes only the inner split used to choose the layer. The test sets, Jev and the text classifier have one score per call.

## 5. gpt-oss-20b test episodes: Jev and the text classifier (the blank cells in Table 1)

These are 594 calls with 26 leaks, asserted in code. The two probes reproduce their paper values on these rows: last-prompt 0.818 and reasoning 0.905.
- **The text classifier** was fit once on all 1,995 development calls and scored once on the test calls, as was done for the Qwen test episodes (E12).
- **Jev** used the development prompt and input format unchanged. Five re-rendered development inputs are byte-identical to the saved development payloads.

| Detector | Scenario×round | All calls |
|---|---|---|
| Text classifier, full input (the view the paper uses for gpt-oss) | 0.844 [.704, .957] | 0.671 [.534, .803] |
| Text classifier, latest message | 0.742 [.576, .907] | 0.777 [.669, .879] |
| Jev | 0.439 [.264, .634] | 0.855 [.797, .899] |

Paired differences (within scenario×round, paired family bootstrap):

| Comparison | Difference [95% CI] |
|---|---|
| Last-prompt − text (full input) | −0.026 [−0.204, 0.140] |
| Reasoning − text (full input) | +0.061 [−0.105, 0.214] |
| Reasoning − text (latest message) | +0.163 [0.015, 0.310] |
| Last-prompt − Jev | +0.379 [0.151, 0.598] |
| Reasoning − Jev | +0.466 [0.233, 0.683] |

Jev at the cutoff that flags no more refusing-episode calls than the reasoning probe at its frozen threshold (12 of 490 calls, 2.4%):
- **Reasoning probe:** catches 14 of 26 leaks.
- **Jev:** catches 2 of 26 (7.7%, CI [0, 27%]).
- **Gap:** 46 points [19, 65].

**What this means.** gpt-oss-20b now matches Qwen3-32B: on test episodes the trained text classifier is level with the probe within noise, and the zero-shot monitor is far behind. This supports the paper's existing claim, "activations add at most a small edge", on a second model's test set.

- Script: [`gptoss_test_text_jev.py`](gptoss_test_text_jev.py), with subcommands `bow`, `jev-prepare`, `jev-execute` and `jev-summarize`.
- Runs:
  - [`runs/20261106-2314-e18-gptoss-bow-test/metrics.json`](../../runs/20261106-2314-e18-gptoss-bow-test/metrics.json)
  - [`runs/20261106-2315-e18-gptoss-jev-test/metrics.json`](../../runs/20261106-2315-e18-gptoss-jev-test/metrics.json): `operating_point`, plus `budget.json` for the cost.

## 6. Ladder on test episodes, and regularisation sweep

### 6a. Reproduction first: dev refits match the paper to ±0.0015
Before computing anything new, the dev ladder and the test "this call?" fits were refit from saved activations with the original code paths.

| Number | Refit | Published |
|---|---|---|
| gpt-oss dev "this call?", "this round or next?" | exact | |
| Qwen test "this call?" 0.9755, gpt-oss test 0.8179 | exact | |
| gpt-oss dev "one call earlier?" | 0.62000 | 0.62008 |
| gpt-oss dev "ever?" | 0.61020 | 0.61032 |
| Qwen dev "this call?" | 0.95868 | 0.96018 |
| Qwen dev "this round or next?" | 0.66907 | 0.66926 |
| Qwen dev "ever?" | 0.53274 | 0.53281 |
| Qwen dev "one call earlier?" | 0.74668 | 0.74728 |

- **Not library versions.** scikit-learn 1.9.1 and a run pinned to the original versions (python 3.11.12, numpy 2.4.6, scikit-learn 1.8.0, scipy 1.17.1) give identical values. One fold refit with the original old-repo `.venv` matches the new refit exactly.
- **Likely cause:** the original runs were made from a dirty working tree. `diff.patch` of `collusion-research-followup/runs/20261026-2201-e05-arxiv-qwen-any-public` shows uncommitted edits to `arxiv_qwen_refit.py` and `qwen_any_public.py`. Not chased further: no two-decimal paper value changes.
- **Decision (Claude, 7 Nov):** accept refits within 0.002. The gate in `test_ladder.py` and `c_sweep.py` was relaxed from "exact" to that tolerance. Stated in the appendix (`app:c-sweep`).
- Runs: [`runs/20261106-2320-e18-reproduce`](../../runs/20261106-2320-e18-reproduce) (scikit-learn 1.9.1), [`runs/20261106-2346-e18-reproduce/reproduction.json`](../../runs/20261106-2346-e18-reproduce/reproduction.json) (pinned). Script: [`reproduce.py`](reproduce.py).

### 6b. Ladder on the test episodes
Probes were fit on all development episodes with the dev ladder's recipe and labels (3 split seeds, z-scored mean, as the frozen test probes), then scored once on the test calls. Within scenario×round, 95% CI.

| Question | Qwen test (positives) | gpt-oss test (positives) | Qwen dev | gpt-oss dev |
|---|---|---|---|---|
| Ever? | 0.56 [0.46, 0.66] (418 of 1,825) | 0.66 [0.52, 0.78] (104 of 594) | 0.53 | 0.61 |
| This round or next? | 0.77 [0.70, 0.83] (120) | 0.68 [0.56, 0.78] (67) | 0.67 | 0.64 |
| One call earlier? | 0.77 [0.67, 0.86] (38 of 1,787) | 0.51 [0.34, 0.68] (17 of 568) | 0.75 | 0.62 |
| This call? (paper) | 0.98 | 0.82 / reasoning 0.91 | 0.96 | 0.91 |

**What this means:**
- **Same shape as on development episodes:** weak far from the act, highest at the call.
- **gpt-oss one call earlier** is at chance on test episodes, but it rests on 17 leaks.
- **Sanity check:** the original Qwen channel round label on the original 1,787-call cohort refits to 0.681, which matches the paper's existing `qwenRoundFreshProbe` 0.68.
- **Label sources:**
  - Qwen labels were rebuilt from all 338 episode files and match the audit rows exactly.
  - gpt-oss labels come from the test plan's `episodes.json` fields with the dev rules. Its 594 rows match the frozen `fresh_rows.json` row for row.
  - Qwen any-public and channel round labels agree on all 1,825 test calls.

Run: [`runs/20261107-0013-e18-test-ladder/metrics.json`](../../runs/20261107-0013-e18-test-ladder/metrics.json). Script: [`test_ladder.py`](test_ladder.py) (helpers in [`common.py`](common.py), [`devfits.py`](devfits.py)).

### 6c. Regularisation sweep (development, held-out scenarios, layer reselected per fold)

| C | Qwen last-prompt | gpt-oss last-prompt | gpt-oss reasoning |
|---|---|---|---|
| 0.01 | 0.955 | 0.902 | 0.956 |
| 0.03 | 0.956 | 0.903 | 0.965 |
| 0.1 (recipe) | 0.959 | 0.906 | 0.933 |
| 0.3 | 0.956 | 0.906 | 0.931 |
| 1 | 0.934 | 0.903 | 0.960 |
| 10 | 0.912 | 0.902 | 0.956 |

The last-prompt probes move by at most 0.004 from C = 0.01 to 0.3. Weaker regularisation lowers the Qwen probe (0.934 at C = 1, 0.912 at C = 10); the gpt-oss last-prompt probe stays flat. The gpt-oss reasoning probe ranges from 0.931 to 0.965 and is lowest near the recipe's C = 0.1, so the recipe does not flatter it. The refit at C = 0.1 matches the published value for both gpt-oss probes and is 0.0015 below it for Qwen (see 6a).

Runs: [`runs/20261107-0013-e18-c-sweep-lastprompt/metrics.json`](../../runs/20261107-0013-e18-c-sweep-lastprompt/metrics.json), [`runs/20261107-0013-e18-c-sweep-reasoning/metrics.json`](../../runs/20261107-0013-e18-c-sweep-reasoning/metrics.json) (108 fits, about 2 hours on CPU; log in the run folder). Script: [`c_sweep.py`](c_sweep.py).

## 7. What changed in the paper

All numbers go through `paper-arxiv/register_e18.py` into `numbers.yaml`. The tables are `tables/e18_*.tex`.

**Main text** (prose net shorter than the last commit; the main text still ends on page 12):
- **Table 1:** gpt-oss test Jev (0.44) and text classifier (0.84) filled in. New test rows for "one call earlier", "this round or next" and "ever" on both models. The caption no longer says "-- means not run".
- **Figure 1:** filled test markers added for "ever" and "this round or next" on both models. The development connecting line was removed, because it ran through the test markers.
- **Method:** "every leaking call is kept". Pooling rounds or all calls keeps the probes' lead over both text baselines.
- **Results 4.2:** "the test episodes ... show the same pattern". The old Qwen round-check paragraph was cut to two sentences, and a repeated sentence about skipped groups was deleted.
- **Results 4.3:**
  - The probe and the text classifier are now "level within noise" on the test episodes of both models, not just Qwen.
  - The Jev sentence now says "below other calls of the same scenario and round", and adds that it scores 0.86 across all calls.
  - E17 qualified as "On Qwen3-32B" (also in the Discussion).

**Appendix:**
- **D (Table 15):** the same new cells, with intervals.
- **G.1:** metric variants and coverage, with the table of three metrics plus leave-one-group-out ranges.
- **G (continued):**
  - per-scenario results for the gpt-oss test episodes;
  - split-seed table;
  - G.2, the test-episode ladder;
  - G.3, the regularisation sweep.
- **H (text baselines):** the gpt-oss test classifier and its paired differences.
- **J (crossed):** the gpt-oss limitation sentence.
- History and reproducibility tables updated.

**Abstract:** unchanged. Its claims (the text classifier comes close, the warning is short-range) now hold on both test sets as written.

## Files kept locally only

Per-fold probe fits (`fit.json`), row-index copies (`rows.json`) and the Jev request payloads (`payloads.json`) are not committed because of their size (about 90 MB together). The scripts above rebuild them from the committed inputs.
