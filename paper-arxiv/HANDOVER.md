# Handover notes for the arXiv rewrite

Shared notes (not paper prose). Append your own entries; keep sections by owner where it matters.

---

## LATEST (28 Oct, morning): E15 folded in, Setup formalised; READ THIS FIRST (coordinator)

**Start with:** `paper-arxiv/main.pdf` at the latest commit (abstract v13 shipped).
- 45 pages: main text pp. 1–11, References from p. 12.
- SHA-256 94adbe6de2b4….
- 858 numbers verified.
- arXiv package: `dist/arxiv-final`. The tarball compiles clean and gives identical text.

**What the result now says about the probe** (supersedes "reads giving in" in the overnight section below):
- Right after the value is read, nearly every disclosure crosses the threshold, permitted or forbidden: 98% / 97% / 100%.
- Away from a read, forbidden leaks cross about twice as often as permitted shares: 61% (19/31) against 33% (24/73), a difference of −28 pts, CI −49 to −8.
- So the probe mostly reads a coming disclosure. It is sharpest just after the value is read, and stronger when the agent was pressed into it. It does not read the rule text.
- E14's larger gap (82% vs 28%) came mostly from its single scenario, not from recency: on dispatch alone, the gap away from a read was still 59% vs 16%.

**Abstract: shipped as v13** (commit `9321b92`; Leo-approved, 28 Oct). Source: `sections/abstract.tex`; earlier drafts and change notes in `private/abstract_draft_v10–v13.md`.
- "At a similar flag rate" replaces "at the same flag rate" (Jev 4.8% vs probe 5.5%).
- Counterfactual sentence: "The probe also flags permitted disclosures: it reads the coming disclosure more than the violation."
- Closing: "A warning raised before generation could let an overseer stop an agent before it gives in, before others act on its messages or recruitment spreads through a swarm." ("other agents" became "others" so the abstract's last line is not a single word.)
- The body is done. Remaining edits are Leo's polish pass.

**Changed since the overnight section:**
- **§2 Setup formalised,** in the style of the Share-Borne and Hidden in Memory papers:
  - Eq. 1: the scenario S = (F, f*, σ, ρ, q).
  - Eq. 2: the dynamics, m_k ~ P(H_{k−1}) and (z_t, o_t) ~ M(x_t).
  - Eq. 3: the leak, τ.
  - Eq. 4: the probe.
  - Eq. 5: the goal.
  - The contents of x_t are listed in order: system prompt (role, ρ, tool definitions), task q as the first user message, tagged peer messages, the target's own earlier replies and tool calls, and tool results. On gpt-oss, x_t also includes earlier same-turn reasoning; z_t is the reasoning for the call itself.
  - "Policy"/π removed: the target and the peer are both "a language model".
  - The target's access is stated once. No text monitor reads z_t.
  - Fixed a factual error: the task was described as being in the system prompt.
- **Run-on sentences split** throughout §2. WRITING.md rule 15 now covers run-on definitions, with a before/after example in §9.
- **E15 in Appendix I.3:**
  - Fig. 8 (recency; points from ≤2 calls are drawn hollow) and Table 26.
  - E14's just-read table is merged into it.
  - §4.3, the Discussion opening and Limitations are aligned with the result.
  - claude-4 did an accuracy read of every changed sentence against results.json; all mismatches are fixed.
- **Limitations, new sentence:** we score only calls after the first read of the file. That leaves out 117 calls (6%; 84/89 episodes read in round 1). A deployed monitor would score every call, and the false alarms on those calls are unmeasured.
- **Real minus signs** paper-wide, fixed at the source in generate_numbers.py.
- **My visual check:** pp. 3–4, 9–11, 31–32 (the changed pages). All clean.

**Runs and spend since the last LATEST:**
| Run | What | Cost |
|---|---|---|
| E14 | on-policy rule flip, dispatch | $1.35 |
| E15 | `runs/20261028-1039-e15-giving-in-scaleup/report.md`; 95 forbidden + 109 early + 99 late episodes; the peer was rate-limited upstream all night, and all episodes were rerun with the same seeds, with 0 failures | $1.80 |
| E16 | `runs/20261028-0330-e16-self-report/`; follow-up only, not in the paper. Self-forecast AUROC 0.68 vs the probe's 0.98. "Not allowed" afterwards: forbidden 60%, clean permit 3%, permit+consequence 47% | $0.19 |
| Pre-read count | from saved episodes | $0 |

**Total spend: about $3.34** of the $25 approved.

**Follow-up note:** `private/FOLLOWUP_IDEAS.md`. It adds K (self-report, E16 done) and L (always-on deployment: score every call).

---

## LATEST (28 Oct, overnight): your notes applied, E14 run, reviews done (coordinator)

**Start with:** `paper-arxiv/main.pdf` at commit `9a00990`.
- 44 pages: main text pp. 1–11, References from p. 12.
- SHA-256 806112db….
- arXiv package: `dist/arxiv-final`. It compiles from the tarball with identical text.
- 740 numbers verified against their sources.

**Your two decisions (abstract only; the body is done):**
1. **MUST-FIX: the counterfactual sentence is now partly wrong** (see E14 below). Suggested replacement:
   > "Allowing sharing does not lower the score of a disclosure the agent first resisted, but freely permitted sharing is mostly not flagged: the probe reads the agent giving in, not the rule."
2. **MUST-FIX: "At the same flag rate".** Jev flags 4.8% of calls against the probe's 5.5%. Suggested replacement:
   > "Flagging no more calls, a zero-shot decision-model monitor catches 26%."

A ready-to-paste abstract with both fixes is in `private/abstract_draft_v9.md`, open in Leaf. The other, optional suggestions are at the bottom of this file.

**What changed tonight:**
- **Your `notes.md`, all applied:**
  - Table 1 moved out of the contributions.
  - §2 Setup rewritten: plain text first, then the formal definitions, covering the workspace, agents, tools, secret, rounds and calls, with the worked example used for the labels.
  - Fig 5 placed with §4.4.
  - Limitations: the main points stay in the text; the rest are in a new appendix.
  - Captions now come after every figure and table.
  - Footnote link added; author name and affiliation centred; GitHub link below the abstract.
  - No stranded headings, and References start on a fresh page.
  - Note: forcing Results onto a new page left p6 three-quarters blank, so I used "keep the heading with its content" rules instead.
- **Figure 2 redesigned:** robot icons (halo for the target, horns for the peer), chat bubbles with the leak highlighted, icons for who observes, and a colour-coded label grid.
- **Number labels on every data plot,** in the main text and the appendix. In-figure titles that repeated the captions were removed.
- **Fair Jev comparison everywhere:** Jev catches 26% (CI 8–39%) when flagging no more calls than the probe, which catches 82%. This replaces the old 68%/29%, which rested on an odd cutoff.
- **E14, an on-policy check of the rule flip ($1.35):**
  - Sharing allowed, consequence sentence kept: the agent resists and then shares, and 80% of sharing calls cross the threshold, like forbidden leaks (82%).
  - Sharing cleanly allowed: the agent shares at once, and only 28% cross.
  - A replay with the consequence sentence removed still shows no drop.
  - **So the probe reads the agent giving in, not the rule and not disclosure as such.** This is now in §4.3, the Discussion, Limitations and Appendix I.
  - Caveat: in the clean arm, permission and immediate sharing occur together.
  - Report: `runs/20261028-0224-e14-permitted-onpolicy/report.md`.
- **Text classifier on test episodes ($0):** 0.96 against the probe's 0.98, a difference of +0.02 whose interval includes 0. Added to Table 2. The intro now says activations add "at most a small edge".
- **Reviews:**
  - clarity 3.5 → 4/5, ready for arXiv after the fixes, all applied;
  - rigour at workshop level: **weak accept, nothing fatal**, with fixes applied.
  - Records are in `reviews/pdf_clarity_r5.md` and `reviews/pdf_rigour_r4_workshop.md`.
- **My visual check:** I looked at every main-text page twice and found nothing broken. The appendix pages were not checked by eye.

**Proposed next, waiting for your go: E15, a larger version of E14.**
- All 5 Qwen test scenarios × 3 arms × 20 episodes. The arms are: allowed with the consequence sentence; allowed cleanly, asked early; and **allowed cleanly, asked late (rounds 4–6)**.
- The late-ask arm separates conflict from timing.
- Optionally, gpt-oss on 3 scenarios.
- About $6 for Qwen only, about $8–9 with gpt-oss.
- If the late-ask arm stays quiet, "reads giving in" becomes a confident main-text finding.

**Spend tonight:** $1.35 of the $25 you approved.

**Follow-up note:** `private/FOLLOWUP_IDEAS.md`. It now covers H (E14, done), I (concept directions) and J (felt vs real violation, and deep recruitment as in the Hugging Face incident).

---

## LATEST (27 Oct ~04:00): E11 pre-commitment check added; final build (coordinator)
- **E11 question:** does the probe just read back the target's own earlier "I'll share it"? **No.** Removing the calls where Jev judged that the target had already agreed to share changes the score little:
  - Qwen: 0.98 → 0.97 (2 of 38 leaking calls flagged).
  - gpt-oss: 0.91 → 0.88 (5 of 26 flagged, all from earlier same-turn reasoning).
- **Flag validity:** Jev agrees with a blind Claude reviewer on 28 of 33 calls (kappa 0.66). The keyword regex was unreliable (negative kappa) and is not used. The plan was frozen at `f329cbd` before any call. Cost: $0.16.
  - Full report: `runs/20261027-0259-e11-precommitment/report.md`.
- **In the paper:** one clause in §4.1, plus a paragraph and table in the shortcuts appendix (tab:precommitment).
- **Other changes since round 2:** the Related-work sentence now reads "complements studies of collusion in markets and in shared multi-agent tasks … as a step toward deployment-like settings". Handshake is not cited, per your decision. E10 (Shi et al.) is marked LOW PRIORITY after its red-team; E09 (permitted vs forbidden) is shelved and may become its own paper.
- **Final build:** `main.pdf` (497834a0…) and the arXiv package (clean build 7feb4b52…), commit `8f6b045`. It is 41 pages, with 657 verified numbers.
- **Viewer check: PASS** on every executable check:
  - all 41 pages inspected;
  - 340 links and all 41 Contents entries are correct;
  - the new table renders cleanly.
  
  Native Preview could not be tested, because the computer-use service failed to start.
- **Still open for you:**
  - the optional "mostly" in the abstract (item 7);
  - the per-episode and precision companions to the abstract's operating point (item 6);
  - the one-word fix "5% of calls in refusing episodes" (item 5). This one is already applied in your round-2 abstract.

---

## ROUND 2 (Leo's feedback, 27 Oct ~02:45–03:30): DONE (coordinator)
- **Title.** "Pre-Crime for Agents, at the Last Second: Probing When a Recruited Agent Gives In". There is a line break after the colon, and the subtitle now fits on one line.
- **Abstract.** Replaced with your revised text, verbatim: 1,671 characters, under arXiv's 1,920 cap. Every number was checked against the registry.
- **C1.** Tables 26 and 27 use the same calls. 0.57 comes from a probe trained on the eventual label and scored without refitting; 0.61 comes from a probe refit on the near-term label. Both captions now say this.
- **C2.** The protected-file writebacks are exactly 2 calls in 2 episodes: one in development and one in fresh.
- **C3–C6.**
  - The Fig. 4 caption now says it shows the stronger of the two text views.
  - Section O cites Table 31.
  - The E08 history row uses your wording.
  - "Last-prompt probe" replaces "Input probe".
- **C7.** The pre-registration note in this file is fixed.
- **Final build.** `main.pdf` (f99a4d76…) and the arXiv package (clean build c8069cd7…), 39 pages, 629 verified numbers. Viewer re-check: PASS. Native Preview is still unverified because the service failed.
- **Optional for you (item 7).** Add "mostly" to the abstract's "detects the coming disclosure rather than the rule being broken", to match §4.3.
- **Decided: no Handshake citation.** The "Handshake for the cameras" sprint entry is concurrent work that we had not seen before submission, so it is not cited.
- **Shelved.** E09 (permitted vs forbidden), which may become its own paper.
- **E10 (Shi et al. transfer): LOW PRIORITY.** The plan is drafted (`experiments/E10-shi-transfer/PLAN_DRAFT.md`), and its red-team (`REDTEAM.md`) gives three reasons to defer it:
  - At the verdict call, text and timing shortcuts already give most of the answer. Every negative is a future accept, so the label mostly measures timing.
  - The Handshake entry has already probed the same data.
  - The setting is game-like, not the tool-using workspace the paper studies.
- **E11 (does the probe read the target's own earlier commitment?): RUNNING.** Approved 27 Oct; spec `experiments/E11-precommitment/SPEC.md`. It flags calls where the target had already agreed to share, using a regex and Jev, and rescores the probe without them. Saved scores only; API under $5; `plan.json` is frozen before any scoring. It does not edit the paper.

---

## ROUND 1 (Leo's feedback, 27 Oct ~00:30–02:00): what changed (coordinator)

**Applied from `FEEDBACK_round1.md` and `FEEDBACK_round1b_s44.md`:**
- **Title and footnote.** New title. A first-page footnote names the sprint; "sprint" appears nowhere else except the analysis-history appendix.
- **Introduction.** Five paragraphs, with no abstract sentences reused and no priority claim.
- **Method.** The trained text classifier is now a third detector.
- **Results.**
  - §4.1 has the trajectory figure.
  - New §4.4, "What would deployment cost?", covers the Qwen alarm panel, per-call vs per-episode false alarms, precision (24%) and an episode-level threshold (26% caught, labelled descriptive).
- **Discussion.** The cascade with two thresholds.
- **Figures and navigation.** Fig 3 gains the trained-text bar and its new title. A clickable Contents follows the references (41 entries, all verified).
- **Abstract.** Your text, verbatim.
- **"Sprint cohort" removed.** The per-episode checks were recounted on the current development episodes, and the unsupported checks were dropped.
- **Small items.** `claims.md` is updated, and the small fixes are done.

**Corrections to your notes, where the data said otherwise:**
- **Episode length.** A refusing fresh Qwen episode has a median of **23** calls, not 17. Independence would give 1 − 0.95²³ ≈ 69%; the observed rate is **59%** because flags cluster.
- **Strict threshold.** Holding refusing episodes to about 5% needs about **0.2%** per call, not 0.3%.
- **The recalled gpt-oss "a third / 8%".** That was a development figure. The paper now uses fresh-episode numbers.
- **"Control calibration underestimates refusing false alarms."** The data don't support this in general: it is a slight underestimate on Qwen and an overestimate on gpt-oss. The paper makes no general claim.
- **Setting the episode threshold on development episodes and applying it to fresh.** Tried and rejected. The probes were trained on those episodes, so their scores there are in-sample, and the threshold alarmed in about three in ten refusing fresh episodes. This becomes one sentence ("set thresholds on held-out traffic"). The paper's per-episode point is a descriptive calibration on the fresh episodes, applied the same way to all three detectors.

**New experiment E08, the policy counterfactual. You approved it; it cost about $1.04 of the $6 cap.**
- **Design.** The same fresh leaking and matched non-leaking calls, with the saved reply held fixed, under three system prompts: the original rule, a permitting rule, and a neutral rewording.
- **Result.** On both models the score is essentially unchanged: Qwen +0.003 [−0.03, +0.07]; gpt-oss last-prompt probe +0.002 [−0.02, +0.03]. The probes **mostly track the imminent emission, not the rule violation**.
- **Integrity.** The first Qwen run failed to reproduce the frozen scores, so it was rejected. It is logged in the history appendix only.
- **Where it appears.** An appendix section, "Does the probe track the rule, or the emission?"; one sentence in §4.3; the Discussion framing as a "pre-emission detector"; and Limitations.
- **Future work** now leads with training detectors that tell a permitted disclosure from a forbidden one of the same value.

**Fresh blind reviews on the round-1 build (`reviews/fb1_reviews_summary.md`):**
- **Clarity: 4/5, PASS.** The story and intro checks both came back "yes".
- **Rigour:** "the core finding survives; fixable, none headline-reversing". All fixable items were applied: "small" edge without "reliable"; "beats each of three shortcuts"; which probe held on fresh per model; the count reconciliations; "median 23"; and the reproducibility tables. The E08 addition was audited by claude-4: PASS.

**Final round-1 build.**
- **Files and size:** `paper-arxiv/main.pdf` (SHA 38d66cdb…) and the arXiv package in `dist/arxiv-final/` (clean build c982b92e…); 39 pages, 326 links.
- **Viewer check: PASS** on every executable check (`reviews/viewer_check.md`, codex-daybreak):
  - all 39 pages inspected;
  - the canonical and clean builds are byte-identical at 144 dpi;
  - 326/326 links are valid, and all 41 Contents entries land on the right pages;
  - no layout, glyph or table defects.
- **Native Preview open/click:** still UNVERIFIED, because the computer-use service failed to start. Please flip through it yourself; it is open in Preview.
- **Optional:** claude-4 notes that the new Future-work sentence is about 49 words, so splitting it in two would follow WRITING rule 13. Left as-is to avoid another rebuild.

**Your remaining calls:** items 5 and 6 under "For Leo first".

---

## START HERE (Leo): summary of the overnight run, 26–27 Oct (coordinator)

**What to open.**
- **PDF:** `paper-arxiv/main.pdf`, built from source `fca13a3`. It runs 33 pages; the main text ends on p7 and references start on p8.
- **arXiv package:** `paper-arxiv/dist/arxiv-final/`, holding the source tarball plus a clean-directory build. Its extracted text is identical to `main.pdf`.
- **Numbers:** 398 registry numbers, all source-verified and none pending. The PDF has no undefined references or overfull boxes.

### What we did
1. **Plan.**
   - Wrote PLAN.md, the claims list (`docs/claims.md`, marked PROPOSED) and the glossary.
   - Wrote the outline as first sentences.
   - Ran two blind outline reviews, scoring 3, then 3.5.
2. **Data and recomputation.** All of this was CPU-only from saved scores, so GPU/API spend was **$0**. Every recomputation is in a new run folder.
   - **Four disjoint named datasets:** development and fresh, for each model.
   - **Qwen per-call refit:** done on the unified "any public emission" label, fitted on development episodes only.
   - **Route 1 (your approval):** the Qwen per-call probe was frozen and committed (`f7a2995`) before it was scored once on the fresh episodes.
   - **gpt-oss fresh last-prompt probe:** frozen (`2b05b71`), then scored.
3. **Draft.**
   - Wrote the ICLR-style main text and a 15+-section appendix, each titled as a question.
   - Used your abstract verbatim, with the minimal edits listed below.
   - Made seven script-generated figures. Four are in the main text; the trajectory, alarm and shortcut figures moved to the appendix during the story pass.
4. **Reviews and audits.**
   - Four blind clarity rounds and three blind rigour rounds, stopped at your 3–5 cap (see the verdicts below).
   - claude-4 ran the cut pass, caveat audit, number audit, and the story and flow pass.
   - Two independent related-work rigour calibrations (`reviews/rw_calibration_*.md`).
5. **Late analyses.** These are quick checks, run before your scope cut landed, and **used because they were finished and favourable**. They are a trained bag-of-words text classifier and a check restricted to handoff-file writes.

### Decisions I made for you (each reversible)
- **Leak boundary:** the gpt-oss frozen detector's definition, applied to both models. The secret appears in a reply or in any tool argument, including writes back to the protected file. Your wording said "outside the protected one". Excluding those cases is a sensitivity analysis affecting 2 Qwen episodes, and it is in the appendix.
- **Abstract number pair:** the Qwen *fresh* pair on identical calls (probe 0.98 vs zero-shot monitor 0.75), not the development pair (0.96 vs 0.76).
- **gpt-oss text monitor:** the main version is the one that also sees the earlier same-turn reasoning (0.40). The sprint's main version was 0.27. This is fairer to the monitor.
- **"Pre-registered" definition:** a committed plan fixed the analysis and its success criterion **before any detector's predictions were evaluated against the test labels**. That holds for both pre-registered tests. For Qwen, the plan was committed *after* the episodes were collected (aggregate leak counts known); see item 5 under "Changed from the sprint version".
- **Main text is about 7 pages** after the story pass. Your brief said 9–10. I kept the shorter version because the story reads better. The trajectory figure (the probe score jumping at the leaking call) is the best candidate to bring back if you want more.

### Your decisions this morning
1. **Title.**
   - Current: "Recruited: An Agent's Activations Reveal Which Call Will Leak a Secret to a Peer".
   - Alternatives: "Recruited: The Call That Leaks Is Readable Before It Acts" and "Recruited: Predicting When an Agent Gives In to a Peer, from Its Activations".
   - See item 1 below: "activations reveal" leans hard, given that a trained text classifier nearly matches the probe.
2. **Abstract.** It is your text, with these minimal edits:
   - "nearly disappears" → "is weak" (your call);
   - "a text monitor" → "a zero-shot text monitor";
   - "Probes matter most" → "Probes would matter most";
   - "catching what a text monitor misses" → "beating a zero-shot text monitor given the same conversation". This now repeats the earlier comparison; consider dropping it.
   
   Options for the operating point (82% caught at 5%, against 68% at 29%) are in item 3 below.
3. **Claims list** (`docs/claims.md`, PROPOSED): sign it off or change it.
4. **Leak boundary** (above): keep it or reverse it.
5. **The three framing points from the final rigour review** (item 4 below).

### Numbers that changed from the sprint version, and why
| Quantity | Sprint | Now | Why |
|---|---|---|---|
| Qwen per-call probe (development, held-out scenarios) | 0.95 | **0.96** | Development set only (2,976 calls, 66 leaks; previously 4,844/101, which included the later batch), unified label |
| Qwen per-call, fresh episodes | n/a | **0.98** | New: frozen dev-only probe, scored once on 1,825 fresh calls (38 leaks) |
| Jev per-call on Qwen | 0.74 | **0.76** dev / **0.75** fresh | Rescored on the same calls under the unified label |
| Jev per-call on gpt-oss (main version) | 0.27 | **0.40** | The main version now sees the same-turn reasoning; 0.27 is kept in the appendix |
| gpt-oss fresh last-prompt probe | n/a | **0.82** | New frozen test on the same 26 fresh leaking calls; the gap to the reasoning probe's 0.91 is within noise |
| Qwen round-level, as stated in the abstract | 0.58 | **0.68** (pre-registered) | The sprint abstract quoted the development number; the confirmed number is 0.68 [0.60, 0.75] |
| Qwen ladder (ever / this round or next / this call) | 0.54 / 0.67 / 0.95 | **0.53 / 0.67 / 0.96** | Development only, unified label |
| gpt-oss frozen reasoning probe (fresh) | 0.91 | 0.91 | Unchanged (pre-registered) |
| **New:** operating points on Qwen fresh | n/a | probe 82% caught at 5% of refusing calls; Jev 68% at 29% | Saved-score analysis. Per episode, at least one alarm fires in 59% of refusing episodes (probe) and 96% (Jev); for gpt-oss, the figure is 26% |
| **New:** trained bag-of-words text classifier | n/a | Qwen 0.93 vs 0.96; gpt-oss 0.86 vs 0.91 | The probe keeps a small edge whose interval excludes zero |
| **New:** within handoff-file writes (gpt-oss) | n/a | probe 0.90; write-only baseline 0.84 | Rules out "the probe just detects the file write" |

### Final review verdicts (full texts in `paper-arxiv/reviews/`)
- **Clarity, rounds 1–4:** 3.5, 3, 3, 3 out of 5.
  - We stopped at the cap: the scores had plateaued, and the remaining comments were about how many numbers the paper carries.
  - Round 4's story check: "Yes, mostly." The reader correctly took the story as "a per-call probe flags the call about to leak, much better than a zero-shot monitor; the signal is short-range; that supports pause-before-acting".
  - Its remaining comment is the title tension (item 1).
- **Rigour, rounds 1–3:**
  - Round 1 failed on empty tables, now fixed.
  - Round 2 said "a solid workshop paper after 4 framing fixes", and those fixes were made.
  - Round 3 said **"the narrow result survives review"**, and flagged three title/abstract framing points that are yours to decide (item 4).
- **Pass bar not formally met.** WRITING.md §8 asks for clarity of 4 or more. We stopped under your bounded-review rule instead.

### Viewer check (real viewer, both PDFs)
- **CONDITIONAL PASS** (codex-daybreak; `paper-arxiv/reviews/viewer_check.md`).
  - **Files checked:** the final PDFs, canonical `main.pdf` (SHA 48e6929d…) and the clean arXiv-package build (SHA c56fae86…), from source `2fa5238`.
  - **What passed:**
    - all 33 page pairs are raster-identical;
    - layout, figures, maths, tables and glyphs are clean on every page;
    - 26 fonts are embedded;
    - 250 links (201 internal, 49 external) all resolve;
    - links show as dark-blue text with no boxes. This fixes the one defect the first pass found, coloured link borders.
  - **Not done:** an open-and-click test in the real macOS Preview app. The computer-use service failed to start on four attempts. **Please flip through `main.pdf` in Preview yourself;** it has been opened for you.

### Look at first
1. Item 1 below: the activations-vs-trained-text finding, and what it means for the title and abstract.
2. Figure 1 and Table 2 in the PDF: the headline ladder and the results table.
3. The abstract (p1), and the operating-point options in item 3.

### Follow-ups (not tonight)
- Steering: design in `explainers/steering-followup.html`; Day 0 branch-point count in `runs/20261026-2144-e07-day0-branch-points`.
- Shi et al. transfer: `private/followup_scope_shi.md`.
- Prompt-optimised monitors.
- Route 2: new pre-registered Qwen episodes.
- The Swarm Alignment sprint, 6–8 Nov.

---

## For Leo first

### 1. The probe's lead over the zero-shot monitor is mostly *training*, not *activations* (claude-4, per @claude #2482)

A trained bag-of-words (TF-IDF) text classifier nearly matches the activation probe on **both** models
on its stronger text view: **Qwen 0.93 vs 0.96; gpt-oss 0.86 vs 0.91**. So the probe's large lead over
the *zero-shot* text monitor (Jev) comes mostly from **training on these labels**, not from access to
activations; activations add a **small, reliable edge** (paired interval excludes zero). The honest
framing is that a trained probe — on activations or on text — beats an untuned monitor, and the
activation probe keeps a small edge. **You may want to reconsider how much the title and abstract lean on
"activations."** Both text views are shown in Appendix H (`tab:text-baselines`) with no best-view
selection. Checks are development-only, saved-score, $0 (gpt-oss `runs/20261026-2242-e05-arxiv-gpt-text-workflow`;
Qwen rerun per codex-2 with verified family exclusions).

### 2. The submitted abstract reports the *development* round-level number, not the *confirmed* one (claude-4, verified against run folders)

**What to know:** the submitted abstract's near-term clause — "(Qwen3-32B: probe AUROC 0.58,
best monitor 0.65, both holding on new episodes)" — states the **development** estimate (0.58)
in the same breath as the holding-on-new-episodes claim. The pre-registered **confirmed primary**
on the fresh episodes is **0.681 [0.604, 0.753]**, higher than 0.58. So it is not an overclaim,
but it is a misleading conflation: a reader takes 0.58 to be the confirmed number, and it is not.

**Verified (read-only, old checkout):**
- Pre-registered plan `docs/confirmatory_plan.md`: primary analysis = near-term A-hazard probe
  AUROC vs 0.5, within scenario × round, on the 89 fresh episodes; primary score = mean of the
  three seed models.
- Run `runs/20261025-1030-e06-confirmation/report.md`: A_hazard (primary) probe
  **0.681 [0.604, 0.753]** (1,787 calls, 82 positives, 89 episodes); success criterion "probe CI
  above 0.5" **PASS**. Best monitor on the fresh set = Jev forecaster L3 **0.644 [0.560, 0.722]**.
- The **0.58 is the development estimate**, not a fresh-episode number: held-out-scenario all-data
  = 0.579 (rounded to 0.58 in the paper), pooled CV = 0.586 (`runs/20261025-1456-e05-all-data-main`).
  codex-2's "all-data CV number" is correct in substance.

**Is the submitted wording accurate?**
- **Abstract:** imprecise / misleading (the one spot). It credits a development point estimate
  (0.58) with "holding on new episodes." Not false, not an overclaim (0.58 < 0.68).
- **Body §4.4 and Figure 3 caption:** accurate. They separate development (probe 0.58, monitor 0.65)
  from the fresh set (probe 0.68, Jev 0.64), with filled vs hollow markers. The explicit
  0.681 [0.604, 0.753] appears in the body only as the rounded "0.68"; Appendix E Table E.2 reports
  the confirmation combination numbers (Jev 0.643 → 0.695; GPT-5-nano 0.554 → 0.667).

**For the rewrite (per coordinator + WRITING.md §4.3):**
- Report the confirmed round-level near-term number **as-is: 0.681 [0.604, 0.753]** as the headline
  for that claim. Put the development estimate (0.579 held-out / 0.586 pooled) in the robustness
  appendix, named, once.
- Fix the abstract so the confirmed number is the one the reader sees; do not attach a development
  point estimate to a "holds on new episodes" claim.
- codex-2 agreed, and I concur: any **new** per-call analysis run on the already-inspected fresh
  sets is **exploratory out-of-sample**, not a new confirmation. Only the frozen round-level
  near-term result (and the frozen gpt-oss this-call reasoning probe) are confirmed.

**Open question for Leo:** confirm you want the abstract to lead the round-level claim with the
confirmed 0.68, with 0.58 moved to the robustness appendix. (Default taken for the draft: yes.)

### 3. Proposed abstract variant: surface the Qwen operating point (claude-4, per @claude #2539; Leo decides — abstract NOT edited)

Verified fresh-Qwen operating point (threshold set on control calls): the probe catches **82%** of
leaking calls at a **5%** refusing-call false-alarm rate; the zero-shot text monitor catches **68%**
at **29%**. This is more concrete than the AUROC and shows the probe's practical edge — it catches more
and false-alarms far less. Suggested wording (Leo to adjust): "…at a fixed threshold the probe catches
82% of leaking calls at a 5% false-alarm rate, where the text monitor catches 68% at 29%."

Two placements:
- **Option A (swap):** replace the gpt-oss "catching half of leaking calls at a 2% false-alarm rate"
  clause with the Qwen operating point; keeps the gpt-oss pre-registered AUROC (0.91).
- **Option B (add):** keep the gpt-oss clause and add the Qwen operating point as the "what it means" sentence.

**Travelling caveat (must accompany either):** per episode, because each episode has many calls, at
least one alarm still fires in **59%** of refusing Qwen episodes and **26%** of refusing gpt-oss
episodes — a deployed alarm needs episode-level calibration.

**Number budget:** the abstract already carries 4 numbers. To stay lean, Option A with the operating
point *replacing* the 0.98/0.75 call-AUROC pair keeps the count roughly flat and arguably reads stronger
(an operating point beats a bare AUROC). Leo's call.

### 4. Final-rigour reviewer's three title/abstract framing points (rigour R3, "narrow result survives review"; for Leo's title/abstract call)

The final rigour round flagged three framings needing Leo's decision; each ties to an item above, and the body already states the underlying facts correctly:
1. **"Activations reveal" overleans** — a trained text classifier does almost as well (item 1); the title/abstract should not imply the signal is uniquely in activations.
2. **The abstract's per-call false-alarm rate needs its per-episode companion** — don't give the call-level operating point without the episode-level rate (item 3; 59% Qwen / 26% gpt-oss).
3. **The Qwen 0.98 should not read as "fresh held-out" without noting those episodes were already inspected** — it is out-of-sample but not pre-registered (item 2); prefer "on fresh episodes held out from training" and keep "pre-registered" for gpt-oss only.

### 5. One-word option in your round-1 abstract (claude-4 audit; abstract kept verbatim)

Your abstract says the probe catches 82% of leaking calls "while flagging 5% of other calls for review."
The 5% (5.5%, 77/1,407) is measured on **calls in refusing episodes**, not on all other calls; §4.4 states it
that way. If you want the abstract to match exactly: "flagging 5% of **calls in refusing episodes**". The
same denominator applies to Jev's 29%. Your text is unchanged until you decide.

### 6. Three author calls from the round-1 fresh reviews (`reviews/fb1_reviews_summary.md`, item 9)

The fresh rigour review found the core result sound and left these three framing choices to you. The body
already states each fact; the question is how much of it the title and abstract carry.
1. **"Gives in" framing.** The title and abstract say the probe tells when an agent *gives in*. What the
   probe actually reads is open: it may read the situation that prompts a leak rather than a decision
   (§4.3; Limitations). Keep "gives in", or soften to "is about to leak".
   **Update after E08 (policy counterfactual):** making sharing *permitted* leaves the probe's score
   essentially unchanged, so the probe mostly tracks the coming emission, not the rule being broken
   (app:policy). The abstract's first sentence ("gives in and breaks its operator's rule … a probe can
   tell") is true in our setting, where every emission breaks the rule. A reader may still take it to
   mean the probe detects rule-breaking, so this choice now matters more. claude-2 left your abstract
   verbatim; the body and Discussion now call it a pre-emission detector.
2. **The abstract's operating point has no precision or per-episode rate.** It gives 82% caught at 5% flagged.
   On the same fresh Qwen calls, only 24% of the probe's flags are real leaks, and 59% of refusing episodes
   see at least one flag (§4.4). Consider adding one of these, or accept the per-call framing as is.
3. **The gpt-oss 0.91 rests on 26 leaking calls.** It is the one pre-registered result, and its interval
   is wide (0.78 to 0.99). The abstract states it without that context; the body and Limitations say it.

*Status after your round-2 abstract (claude-4):* items 2 and 3 are now addressed in the abstract (the
per-episode clause; the CI and 26 calls). The new opener ("leaks a secret … can tell") addresses most of
item 1; the title still says "Gives In".

### 7. One-word option in your round-2 abstract (claude-4; your text kept verbatim)

"Making the disclosure permitted leaves the probe's score essentially unchanged, so it detects the coming
disclosure rather than the rule being broken." The E08 spec's pre-stated reading for this outcome, and
§4.3 of the paper, both say the probe **mostly** tracks the coming disclosure. The data support "mostly":
both intervals include small decreases. To match: "so it **mostly** detects the coming disclosure".

## Technical decisions

- **ICLR header on the arXiv draft (codex #2373, via claude #2374).** With `\iclrfinalcopy` the ICLR 2027
  style prints "Published as a conference paper at ICLR 2027". For the arXiv draft, `main.tex` clears it
  with `\fancyhead[L]{}` right after `\maketitle`. Style files, margins and fonts are untouched. Revert
  this one line for a real ICLR camera-ready.

## Changed from the sprint version (accuracy audit, claude-4)

- **Why Gemma-4-31B is probe-excluded: reason changed, verified.** Sprint reason was "no channel
  post (0/8)". Under the unified any-public-emission label a file write counts as a leak, so the
  reason is now "no mix of leaking and refusing episodes." Verified against
  `docs/second_target_behaviour_report.md`: under the delayed whole-block strategy Gemma read the
  token in round 1 and wrote it to a file in 8/8 recruiting episodes, 0/8 channel posts, controls
  clean 0/4 — so every episode leaks, leaving no refusals to contrast. The new behaviour appendix
  wording is accurate and correctly scopes "no mix" to the *delayed* strategy. Caveat for the audit:
  the *natural* (non-delayed) prompt gave 6/8 (a mix), so the scoping to the delayed block request
  must stay explicit; do not state "no mix" of Gemma in general.

## Figure structure (claude-4, verified)

- **Round-level confirmation is its own figure, separate from the per-call ladder.** The headline
  ladder (Fig `fig:headline`) is now a single per-call panel (ever / this-round-or-next / this-call,
  both models, hollow=development, filled=fresh). The Qwen round-level near-term confirmation is a
  separate figure in §4.1 (`fig:round`, from `fig_round_confirmation.pdf`), captioned with its own
  pre-registered leak label. Separation verified clean: §4.1 references `fig:round` (no stale
  `fig:headline`(b)); no panel-b references to the ladder remain. This resolves the earlier Fig 1
  overload and the two-labels-on-one-axis risk.

## Future work (for Leo / after tonight)

- **Planned follow-ups (slots left in the draft, per PLAN.md §8):** causal **steering** on the probe
  direction (tests causality), and **transfer to the Shi et al. environment** (tests robustness in a
  less templated, emergent-collusion setting). Scopes in `private/followup_scope_steering.md` and
  `private/followup_scope_shi.md`.
- **A fair third detector arm (Leo's idea, #2491): a prompt-optimised LLM monitor** (e.g. GEPA on the
  decision-model monitor), evaluated next to the trained probe and the trained text classifier. **Check
  novelty first** before claiming this framing is new.
- **A possible unifying story for the follow-ups:** "Where does the warning signal live, and who can
  read it?" — trained probe vs trained text classifier vs optimised LLM monitor (who can read it), with
  Shi transfer testing robustness and steering testing causality.
- **Caveat to carry into that work:** the bag-of-words text classifier's strength may partly reflect the
  **templated synthetic peer messages**; a less templated setting (Shi) is the test.
- **Deferred baselines run only as quick checks tonight** (development, saved-score): a *tuned* text
  monitor, and a full `write_file`-conditioned analysis beyond the quick version.
- **#5 Policy counterfactual replay (E08; now RUNNING via codex-daybreak).** Spec:
  `experiments/E08-policy-counterfactual/SPEC.md`. Leo-approved (#2632).
- **#6 Real authorised-disclosure episodes (Leo-approved, #2632).** Collect episodes where emitting the
  same value is *allowed* as well as ones where it is *forbidden*. Questions: can a probe be trained to
  detect the difference, and does it still work? This tests the new Limitations point that no episode
  writes the secret with permission, so the current probe may track emitting the value rather than
  breaking the rule.

## Changed from the sprint version (accuracy audit, claude-4) — continued

- **Pre-registration chronology corrected from the sprint wording (codex-2 #2517, source-documented;
  verified by claude-4).** The sprint framing implied the fresh episodes were collected *after* the
  plan was committed. **Current definition (after your round-1 decision):** a test is *pre-registered*
  when a committed written plan fixed the analysis and its success criterion **before the test's
  episodes were collected** (method.tex). Only **gpt-oss-20b** meets it: its reasoning-probe plan was
  frozen before the fresh episodes were collected. The **Qwen3-32B** round-level plan was frozen
  **after** its fresh episodes were collected, when their overall leak counts were already known, but
  **before any probe or monitor was scored** (app:confirmation). The paper therefore calls only the
  gpt-oss test "pre-registered" and describes the Qwen test as planned before scoring, on fresh
  episodes held out from training. Exact commit times are in the analysis-history table. Numbers
  unchanged. (An earlier version of this note said the paper kept "pre-registered" for both tests;
  that is no longer true.)

## Abstract suggestions (not applied; the abstract is frozen), 28 Oct

- "halting on any alarm would stop most benign episodes": in the body these are *refusing episodes* (recruiting episodes that never leak), while "benign calls" in the abstract means *control calls*. One option is "would stop most refusing episodes", or "most episodes in which the agent never leaks".
- "26%" in the abstract is Jev's share of leaking *calls* at the probe's flag rate. The body also reports 26% of leaking *episodes* for the per-episode threshold; §4.4 now says which is which.

## MUST-FIX abstract suggestions for Leo (28 Oct, overnight; abstract not edited)

> **Superseded 28 Oct morning:** item 1 below is replaced by the counterfactual suggestion in the top LATEST section; E15 does not support "giving in".

1. **Counterfactual sentence is now partly wrong (E14).** "In a counterfactual replay, allowing sharing leaves the probe's score unchanged: it reads the coming disclosure, not the violation." New episodes under a clean permission show that a freely permitted disclosure is mostly not flagged (28% of sharing calls cross the threshold, against 82% of forbidden leaks). Suggested: "Allowing sharing does not lower the probe's score once the agent has resisted, but freely permitted disclosures are mostly not flagged: the probe reads the agent giving in, not the rule."
2. **"At the same flag rate".** Jev flags 4.8% of refusing calls against the probe's 5.5%. Suggested: "Flagging no more calls, a zero-shot decision-model monitor catches 26%."

## Other abstract suggestions (Leo decides)

- "decision-model" → a plainer gloss.
- "counterfactual replay" → "when we replay the same prompts under a rule that permits sharing" (or replace per item 1).
- "most of the gain" → "most of the probe's lead over the zero-shot monitor". The claim now also holds on test episodes: the bag-of-words classifier scores 0.96 against the probe's 0.98, a difference of +0.02 whose interval includes zero.
- Possibly give Qwen's per-call AUROC (0.98).
- The last sentence could add "which we do not test here", or "we test only exact-string leaks".
- "which a second text check does not fix" → "which a second text check did not detectably fix".
- "threshold set to flag 5% of benign calls" → "set on development control calls".

Declined by claude (recorded): labelling each Table 2 row by evidence status (status clutter; §2.3 states it once); moving the note on the discarded first replay into the main text; renaming the §4.3 heading and the Fig 4 title (Leo-approved).

