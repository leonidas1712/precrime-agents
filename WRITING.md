# Writing guide for this paper

This file is for every agent that writes or edits text in this repository: paper sections, the appendix, abstracts, figure captions, READMEs and blog posts. Read all of it before writing anything. It takes priority over your default writing habits.

**It applies to every part of the paper: main text, appendix, captions, tables and footnotes.** The appendix is not a dumping ground. It may hold more detail, but it follows the same rules (see §5).

## 1. Why this guide exists

The sprint version of this paper was correct and rigorous, and hard to read. Readers who reviewed it cold, human and automated, all said the same thing: it was confusing. Nobody said it was wrong.

The cause was not any single bad sentence. Each added caveat, extra number and new label was locally justified. Each one guarded a real claim against a real objection. Added together, they buried the findings.

You will feel the same pull. When you want to add a qualifier, a second number or a parenthetical, assume this is that pull. Follow the rules below rather than your instinct.

**Rigour lives in three places:**
- choosing claims that are true and stating them precisely,
- the experiments and controls,
- a complete, well-organised appendix.

**Rigour does not live in hedging density.** A claim worded precisely once is more rigorous than a vague claim with five qualifiers attached.

## 2. Who we are writing for

There are two readers.

- **Main-text reader.** A competent ML engineer or researcher who does not work on probes or multi-agent collusion. They read the abstract, look at the figures, and read the main text once, quickly. They must come away knowing the question, the setup, the three main findings, and the biggest limitation.
- **Appendix reader.** A careful reviewer checking whether the claims hold up. They want every dataset, count, seed, control and caveat, and they want to find each one quickly.

The main text serves the first reader. The appendix serves the second. When the two conflict, move the detail to the appendix and point to it once. Both readers deserve clear writing.

## 3. Rules for all text

These are checkable. An editor should be able to mark each one pass or fail.

### Claims
1. **Every paragraph opens with its claim, in one plain sentence.** The rest of the paragraph supports it. A reader skimming only first sentences should get the paper.
2. **Every paragraph in Results maps to an entry in the claims list (§7).** If it serves no claim, cut it or move it to the appendix.
3. **Show evidence status by which data a number comes from, never by status words.** See §4.4.

### Numbers
4. **At most one number per claim in the main text.** Put the headline number in the sentence. Put supporting numbers in a figure, a table or the appendix.
5. **No confidence intervals in main-text prose.** Give point estimates and link to the appendix section that prints the intervals. An exception needs a stated reason. Figures may show whiskers.
6. **At most three numbers in the abstract.**
7. **Round to two decimal places.** Use percentages for rates.
8. **Do not report counts in prose.** Refer to datasets by name (§4). Counts live in one table.

### Words
9. **Fixed vocabulary.** Define each technical term once, in plain words, the first time it appears. Then use exactly that term every time. No synonyms, no variants, no abbreviations introduced later. Keep the list in `docs/glossary.md`.
10. **No more than about six defined terms in the main text.** If you need a seventh, drop or merge one.
11. **No invented terms when an ordinary phrase works.** Write "the call that leaks", not a coined label.
12. **Name things by what they are, not by internal codes.** No experiment IDs, run folders, label letters (A, B, C), access-level codes (L1–L4) or file names in the main text. In the appendix, these appear only in the reproducibility index (§5).
12a. **Banned padding words (author's request, 30 Oct 2026).** Never write "frozen", "freeze", "pre-registered", "preregistered", "exploratory", "post hoc", "confirmatory", "planned in advance" or "fixed in advance" in the main text, and this includes figure labels and captions. State once, in Method, what was fixed before the test data were collected, then report results plainly ("the probe scores AUROC 0.91 on the test episodes"). The details, including the commit, belong in one appendix section.

### Sentences
13. **Keep sentences under about 30 words.** Split anything longer.
14. **At most one parenthesis per sentence.** Never nest them.
15. **No chains of semicolons, and no run-on definitions.** Use separate sentences. This matters most when introducing notation: give each object its own sentence, rather than packing several facts about several objects into one sentence joined by "and", "which" and ";". See §9, "Defining notation".
16. **Use active voice with a concrete subject.** "The probe catches 85% of leaking calls", not "85% of leaking calls are caught".
17. **Do not narrate the history of the analysis.** Describe what each analysis is, not when it was decided. Analysis history lives in one appendix section (§4).

### Caveats
18. **Inline caveat rule.** A caveat may appear inline only if the sentence would be false or misleading without it. Otherwise it goes in Limitations.
19. **Each limitation is stated once, in one place.** Do not repeat it in the abstract, results and discussion.
20. **Limitations are ordered by how much each could change a headline,** with one short paragraph each.

### Figures and tables
21. **Each figure answers one question, and its title states that question or its answer.** Example: "The nearer the act, the easier to predict."
22. **Main-text captions are at most three sentences.** Extended detail goes in an appendix note referenced from the caption.
23. **A reader should get a figure's main result without reading its caption.**
24. **Prefer figures to tables in the main text.** A main-text table is allowed only when the reader genuinely needs to look up values. Tables add cognitive load.

## 4. Datasets, confirmation sets and post-hoc analyses

This is the problem that produced most of the hedging in the sprint version. The paper had many overlapping call sets, two confirmation sets, label definitions that differed between models, and analyses designed after seeing earlier results. Agents tried to carry all of that provenance inside every sentence. Do not do this. Handle it with the structure below.

### 4.1 Name a small number of datasets, once
Define the datasets in the Setup section, in one short paragraph plus one figure or small table. Give each a plain name. For example:
- the **development episodes** (everything collected during the sprint),
- the **test episodes** (collected after the analysis choices were written down), one set per model.

Use at most three or four named datasets in the main text. Every number in the paper belongs to exactly one of them. Refer to them only by name. All counts go in a single table, in the main text if small or in the appendix.

### 4.2 Fix one evaluation per question
For each question in the paper, choose one evaluation as the reported one. A sensible default is held-out scenarios on the development episodes, plus the fresh test episodes where they exist. Every other evaluation, such as pooled cross-validation or fold-stratified scoring, goes in an appendix robustness section. Never report two evaluations of the same thing side by side in the main text.

### 4.3 Remove mismatches instead of explaining them
If two numbers in the main text come from slightly different call sets, the fix is to recompute one of them, not to add a caveat. Example: the sprint paper reported the Qwen near-term result as 0.58 in one place and 0.67 in another, because the second set included the leaking calls. A reader cannot hold that distinction. Pick one call set per question and recompute so that every comparison in a figure uses the same calls.

The same goes for label definitions. If the two models use different definitions of a leak, either recompute one model under the other's definition, or define the difference once in Setup and never mention it again in the main text.

Rules for recomputation:
- Recompute from saved scores and activations. This is reanalysis, not new experiments.
- Save every recomputation as a new run folder. Never overwrite an existing result.
- Never change a result that was scored under a committed plan. If a confirmation set was scored on a specific call set, report that number as it was.
- Record every recomputed number, and what it replaced, in the analysis-history appendix.

### 4.4 Show evidence status by structure, not words
The main text never labels results with status words ("frozen", "pre-registered", "exploratory", "post hoc", "confirmatory"); see rule 12a. Readers learn a result's standing from **which data it comes from**, which the text states plainly ("on the test episodes", "on the development episodes"), and from:
1. **One procedural sentence in Method** saying what was decided before the test data were collected, e.g. "Before collecting the gpt-oss-20b test episodes, we committed the analysis plan, threshold and success criterion to the repository (App F)."
2. **A visual convention in figures,** explained once in the first results caption, e.g. filled markers for test episodes and hollow markers for development episodes.
3. **The analysis-history appendix (§4.6),** which says, once and completely, which analyses were planned before which data and which were added after seeing results.

The claims list may keep an internal status field; it never reaches the prose.

### 4.5 Keep provenance in the numbers registry, not in sentences
Agents hedge partly because inline qualifiers are the only place they can record where a number came from. Give provenance a proper home instead.

Keep `paper-arxiv/numbers.yaml`, with one entry per number used in the paper:

```yaml
qwen_percall_probe_auroc:
  value: 0.95
  ci: [0.93, 0.97]
  dataset: development episodes
  evaluation: held-out scenarios
  dataset_role: development   # internal only; never printed
  run: runs/20261026-1237-e05-qwen-percall
```

Generate `paper-arxiv/numbers.tex` from it, one LaTeX macro per entry, for example `\newcommand{\qwenPercallAUROC}{0.95}`. Write every number in the paper through its macro, never by hand. Provenance is then recorded once, every number is consistent everywhere it appears, and the text is free to state the claim plainly.

### 4.6 Analysis history goes in one appendix section
Write one appendix section titled "Analysis history." It contains a short table listing each analysis, whether it was planned before data collection, added after seeing which result, and whether it was later confirmed. This is where honesty about the research process lives. It is stated once, completely, and nowhere else.

## 5. Appendix rules

The appendix follows every rule in §3. In addition:

1. **Organise it by question, not by experiment order.** Each appendix section's title is a question, such as "Does the probe just detect the read step?"
2. **Each section opens with its answer in one or two sentences.** The supporting detail follows.
3. **One table per question,** rather than paragraphs listing many numbers.
4. **No duplication.** If something is in the main text, the appendix extends it and does not restate it.
5. **Run paths and file names appear only in a final "Reproducibility index" section,** as a table mapping each figure, table and headline number to its run folder and script. Elsewhere, refer to results by name.
6. **The appendix may carry more numbers and intervals than the main text,** but still in tables, not prose.

## 6. LaTeX setup (ICLR 2027 style)

We use the official ICLR 2027 style files, de-anonymised, for an arXiv preprint. Keeping the official style means the same source can later go to an ICLR or NeurIPS workshop with minimal changes.

### Files
Work in `paper-arxiv/`:

```
paper-arxiv/
  main.tex               % preamble, \input of sections, bibliography
  iclr2027_conference.sty, iclr2027_conference.bst, fancyhdr.sty, natbib.sty
  math_commands.tex      % from the template; add our notation here
  numbers.yaml           % numbers registry (§4.5)
  numbers.tex            % generated macros; never edit by hand
  sections/              % one file per main section
  appendix/              % one file per appendix section
  figures/               % vector PDFs produced by scripts
  figures/src/           % the scripts that produce them
  references.bib
```

### Required settings
- Put `\iclrfinalcopy` before `\begin{document}`. This shows the author name and removes line numbers and the "Under review" header.
- **Do not modify the style files,** margins, font sizes or spacing. Use no `\vspace` hacks to squeeze content. If it does not fit, cut words.
- Use `natbib` citations: `\citet{}` when the authors are part of the sentence, `\citep{}` otherwise.
- **Every bibliography entry must be checked against its live source before release:** it must exist, and its title, authors (order and spelling), year and ID must match. Each must support the claim it is cited for.
- **Every entry needs a working link:** a `url` field (`https://arxiv.org/abs/<id>` for arXiv papers) or a DOI. An arXiv ID alone prints without a link.
- **Date arXiv-only papers by their first arXiv version.** For a published version, give the venue and its year.
- **Never write a venue status you cannot verify,** such as "under review at X". Write "Unpublished manuscript" instead.
- **Before release, delete uncited entries,** and check that no month field prints as a raw number.
- Use `booktabs` for tables (`\toprule`, `\midrule`, `\bottomrule`), with no vertical rules.
- Use `cleveref` (`\cref{}`) for references to figures, tables, sections and appendices.
- Start the appendix with `\appendix` after the references.
- Put the LLM usage statement in an unnumbered section after the conclusion.

### Figures
- Produce every figure from a script in `figures/src/`, saved as vector PDF. Never hand-edit a figure.
- Size figures with `width=\linewidth` (or a fraction of it). Keep font sizes in figures close to the body text size.
- Use one colour-blind-safe palette across all figures. Each model, and each detector type, keeps the same colour everywhere.
- **Put a headline figure on page 1 or 2.** It should show the main result at a glance, so a reader who sees only that figure gets the paper.

### Length
- Aim for **9–10 pages of main text**. arXiv has no limit, but this keeps us honest about brevity and makes a later workshop submission easy.
- References and appendix are unlimited, but they still follow §3 and §5.

### Notation and formal definitions
Short formal definitions are welcome when they make things clearer than prose. Use them for the objects the paper is built on: episodes, calls, labels, the probe and the metric. Define notation once in a short "Problem setup" subsection, add the macros to `math_commands.tex`, and keep the number of symbols small.

An example of the right scale:

> An episode is a sequence of target calls $c_1, \dots, c_T$. Let $\tau$ be the index of the first call that leaks the secret, with $\tau = \infty$ if none does. For each call $c_t$ before or at the leak, we define three labels by distance from the act:
> $y^{\text{call}}_t = \mathbb{1}[t = \tau]$ (this call leaks),
> $y^{\text{near}}_t = \mathbb{1}[\text{the leak falls in this round or the next}]$, and
> $y^{\text{ever}}_t = \mathbb{1}[\tau < \infty]$ (this episode ever leaks).
> A probe reads the residual-stream activation $h_\ell(c_t)$ at layer $\ell$, at the last prompt token, and outputs a score $s_t = w^\top h_\ell(c_t) + b$.

Do not formalise things that are clearer in words. Do not introduce symbols that are used only once.

### Building and checking
- Build with `latexmk -pdf main.tex`. The build must finish with no undefined references or citations.
- For arXiv, upload the source with the generated `main.bbl` included.
- Before handing over a draft, open the PDF and check every page visually: figure placement, overfull lines, broken references.
- **A clean CLI build is not enough.** An agent with computer use must open the compiled PDF in a real viewer, such as macOS Preview, and look at it the way a reader would. A Codex-harness agent with computer use is ideal for this. It checks that:
  - the file opens without errors or warnings,
  - fonts render correctly, with no missing glyphs or odd substitutions,
  - every figure appears and is sharp at normal zoom,
  - math, tables and hyperlinks display correctly, and cross-references and citations are clickable,
  - nothing runs off the page or overlaps.
- Do the same viewer check on the arXiv build. Package the source as it would be uploaded, compile it from that package in a clean directory, and open the result. This catches missing files, a missing `.bbl` and figures that only worked locally.

## 7. Process

Follow these steps in order.

**Step 1. Claims list.** Write `docs/claims.md` with three to five claims. For each claim, record:
- the claim in one plain sentence,
- its headline number, as a `numbers.yaml` key,
- which data it rests on (test or development episodes; internal note, never printed as a status word),
- the dataset and evaluation it uses,
- the figure that shows it,
- the one caveat that must travel with it, if any.

Get human sign-off. If the human is unavailable, mark the list "proposed", continue, and flag it at handover. Every later paragraph must serve one of these claims.

**Step 2. Outline as first sentences.** Write the paper as a list of paragraph-opening sentences only, section by section, including the appendix sections as questions with one-line answers. Someone reading just this list should understand the paper.

**Step 3. Blind read of the outline.** Spawn a fresh subagent with no access to the repo, this conversation or the claims list. Give it only the outline. Ask it the questions in §8. If its answers do not match the claims list, fix the outline before drafting.

**Step 4. Draft.** Fill each paragraph under its opening sentence. Write the main text first, then the appendix. Put every number through `numbers.tex`.

**Step 5. Cut pass.** A separate editor agent goes through the text with one job: make it shorter and plainer without changing what it claims. The editor may delete, move content to the appendix, split sentences and replace words. The editor may not add content, except to fix something false. It reports what it cut and where it moved it.

**Step 6. Caveat audit.** List every hedge, qualifier and parenthetical in the main text. For each, answer "Is the sentence false or misleading without it?" If no, move it to Limitations or the appendix.

**Step 7. Number audit.** Check that:
- every number in the PDF comes from a `numbers.tex` macro,
- every registry entry points to an existing run folder and matches it,
- no main-text comparison mixes datasets or evaluations.

**Step 8. Two blind reviews of the draft.** Spawn two new fresh subagents with no prior context, each given only the compiled PDF:
- a **clarity reader** (§8, questions 1–8),
- a **rigour reviewer** (§8, questions 9–12).

Repeat after every major revision. Both must pass.

**When a reviewer, human or agent, raises a concern, fix it in this order of preference:**
1. Reword the claim so it is precisely true.
2. Add or extend an appendix section.
3. Add an item to Limitations.
4. Only as a last resort, add an inline caveat, and only under rule 18.

### Do not game the blind reviews
The reviews are a signal, not the goal. These moves raise a reviewer's verdict while making the paper worse, and are not allowed:
- adding friendly or flattering language, enthusiasm, or pleasantries,
- overstating results, or dropping a caveat that rule 18 requires,
- telling the reviewer what you want to hear, priming it, or showing it earlier versions,
- removing hard-but-true content to make a section feel easier.

The target is a paper that a careful reader finds clear and a sceptical reader finds sound. A clarity verdict bought at the cost of accuracy is a failure.

## 8. Blind-review questions

Give each reviewer only the text or PDF under review, with no other context.

**Clarity reader**
1. In one sentence, what question does this paper ask?
2. In two sentences, what is the setup?
3. What are the main findings, with their key numbers?
4. What is the single biggest limitation?
5. Which sentence or paragraph was hardest to follow, and why?
6. List every term you had to guess the meaning of.
7. Was there any point where you lost track of which set of results a number belonged to?
8. On a scale of 1–5, how easy and pleasant was this to read, and what one change would most improve it?

**Rigour reviewer**
9. Which claims are not adequately supported by the evidence shown?
10. Which claims are stated more strongly than the evidence allows?
11. What important caveat is missing or hard to find?
12. Could a reader reproduce the main results from the paper and appendix? What is missing?

**Pass** means:
- answers 1–4 match the claims list,
- answers 5–7 turn up nothing serious,
- the score in 8 is 4 or 5,
- 9–11 turn up nothing that changes a headline.

Anything listed under 6 or 7 is a defect to fix, not a matter of taste.

## 9. Before and after

These examples are from the sprint version.

**Abstract sentence**

> Before: In follow-up experiments asking whether this call will give in, a probe reading before the call generates anything is strongly discriminative: AUROC 0.95 on held-out scenarios for Qwen3-32B and 0.91 for gpt-oss-20b, above a Jev decision-model monitor given the same conversation (0.74 and 0.27).

> After: A linear probe on the model's internal activations can tell which call is about to leak, before that call writes anything (AUROC 0.95 on Qwen3-32B). A text monitor shown the same conversation does worse.

What changed:
- Four numbers became one.
- The second model's number moved to Results.
- The monitor's name and scores moved out of the abstract.
- "Strongly discriminative" became what the probe actually does.

**Describing the data**

> Before: The round-level cohort is 4,697 eligible recruiting calls from 241 episodes, grouped into families of related episodes (variants of one seeded episode and reruns that share its history; §3.5). The per-call cohort, which also includes the give-in calls, has 4,844 calls (101 give-ins; L.3).

> After: We score every call the target makes after it reads the secret, up to and including the call that leaks it. We call these the development episodes. Table 1 gives the counts.

**Results paragraph with an inline caveat**

> Before: These post-hoc rungs include give-in calls, so Qwen3-32B near-term reads 0.67 here, not the confirmed 0.58 (Appendix L.3).

> After: Recompute so both numbers use the same calls (§4.3), then delete the sentence.

**A monitor result that needs explaining**

> Before: Jev, asked the same question, scores lower on both models: 0.74 on Qwen3-32B and 0.27 on gpt-oss-20b (probe minus Jev +0.21 and +0.64, intervals exclude zero). Jev sees the same conversation and tool results, with protected values masked; for gpt-oss-20b it omits same-turn reasoning; Jev given that reasoning scores 0.40 (Appendix L.3). On gpt-oss-20b it flags nearly every pressured call, so it ranks leaking calls below the rest (Appendix L.3).

> After: A text monitor given the same conversation scores lower on both models. On gpt-oss-20b it flags almost every call made under pressure, so it cannot separate the calls that leak from those that do not. Appendix C compares the two in full.

**Defining notation (run-on sentence)**

> Before: The target is a language model $M$ whose system prompt states its role and $\rho$ and which receives $q$ as its first user message, and the peer is a second language model $P$ that writes each message from the channel so far; the target reaches $F$ only through its tools, and the peer sees only the channel, never $F$, $f^\star$ or $\rho$.

> After: The target is a language model $M$. Its system prompt states its role and $\rho$, and it receives $q$ as its first user message. It reaches $F$ only through its tools. The peer is a second language model $P$ that writes each message from the channel so far, and it never sees $F$, $f^\star$ or $\rho$.

What changed:
- One 60-word sentence became four short ones.
- Each sentence is about one object: the target, its inputs, its access, the peer.
- The semicolon and the "whose … and which … and …" chain are gone.

## 10. Final checklist

Before calling any draft done, confirm:

- [ ] The claims list exists, and is signed off or flagged as proposed.
- [ ] Every Results paragraph opens with its claim and maps to the claims list.
- [ ] The abstract has at most three numbers and follows the template in §11.
- [ ] There is at most one number per claim in the main text.
- [ ] The main text uses at most four named datasets, and every number belongs to one of them.
- [ ] No main-text comparison mixes datasets, evaluations or label definitions.
- [ ] Evidence status is shown by structure and figure markers, not repeated in sentences.
- [ ] Every number comes from `numbers.tex`, and every registry entry traces to a run folder.
- [ ] No internal codes or run paths appear outside the reproducibility index.
- [ ] Every defined term is used consistently and is listed in the glossary.
- [ ] Every inline caveat passes rule 18. Each limitation appears once.
- [ ] The appendix is organised by question, each section opens with its answer, and nothing is duplicated.
- [ ] Main text is 9–10 pages, with a headline figure on page 1 or 2.
- [ ] The PDF builds cleanly and was checked page by page.
- [ ] An agent with computer use opened both the local PDF and the packaged arXiv build in a real viewer, and both render correctly.
- [ ] The latest clarity and rigour reviews both passed (§8).

## 11. Paper structure

| Section | Job | Length guide |
|---|---|---|
| Abstract | Problem, setup, main finding, what it means, biggest limitation | 150–200 words, ≤3 numbers |
| 1. Introduction | Why this matters, the question, the answer in one paragraph, a headline figure, contributions as a short list | 1.5 pages |
| 2. Setup | The scenario, the roles, what counts as a leak, the named datasets, short formal definitions, one worked example | 1.5 pages |
| 3. Method | Probes and text monitors in plain terms, how they are compared fairly, how results are confirmed | 1 page |
| 4. Results | One subsection per claim, test-episode results first | 3–4 pages |
| 5. Related work | Where this sits, in contrasts rather than lists | 0.5–1 page |
| 6. Discussion and limitations | What it means for oversight, where it does not apply, limitations ordered by impact | 1 page |
| Appendix | Organised by question, ending with the analysis history and the reproducibility index | Unlimited |

**Abstract template:**
1. One to two sentences: the problem and why it matters.
2. One to two sentences: what we built and measured.
3. Two to three sentences: the main findings, with at most three numbers.
4. One sentence: what this means for oversight.
5. One sentence: the biggest limitation or next step.
