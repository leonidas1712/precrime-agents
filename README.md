# Pre-Crime for Agents, at the Last Second

**Probing when a recruited agent gives in**

Leonidas Raghav · 2026

[**Paper (PDF)**](paper-arxiv/main.pdf) · **arXiv:** *link coming soon* · [**Citation**](#citation)

---

When AI agents share channels and credentials, one agent can press another to break its operator's rule, a first step towards collusion. Output filters catch a leak once it is written, but by then other agents may have read it. This repository asks whether an overseer can tell **beforehand** that an agent is about to give in.

We find that a linear probe on the agent's activations can, in a controlled multi-agent setting. Read at each model call *before generation*, the probe picks out the call that leaks.

<p align="center">
  <img src="assets/readme/fig2_setup.png" width="100%" alt="How an episode works: a real leak, the agents and what each detector observes, and the three labels">
</p>
<p align="center"><em><b>How an episode works.</b> (a) A real leak on Qwen3-32B, token redacted. (b) The peer asks over the team channel for a token the target's operator told it to keep. The text monitor reads the transcript; the probe reads the target's activations before each output. (c) The three labels, from "will this episode ever leak?" to "will this call leak?".</em></p>

## Main results

**1. The call that leaks is predictable just before it acts, and this held on test episodes.**
- **Qwen3-32B:** on held-out test episodes, the last-prompt probe (fit on development episodes only, then frozen) scores AUROC **0.98**. At a threshold set to flag 5% of benign calls, it catches **82%** of leaking calls (31 of 38). At a similar flag rate, a zero-shot decision-model monitor catches **26%**.
- **gpt-oss-20b:** a probe on the model's reasoning scores AUROC **0.91** in a pre-registered test on new episodes. At its frozen threshold it catches 14 of 26 leaking calls while flagging 2.4% of calls in refusing episodes.

**2. The warning comes late.** Asked three questions about the same calls, the probe is accurate for "will *this call* leak?", modest for "this round or the next", and near chance for "will this episode *ever* leak?".

<p align="center">
  <img src="assets/readme/fig1_headline.png" width="85%" alt="AUROC rises from 'ever' to 'this round or next' to 'this call' for both models">
</p>
<p align="center"><em>The nearer the act, the more predictable the leak. Hollow markers: held-out scenarios on development episodes. Filled markers: test episodes, each scored once.</em></p>

**3. The score jumps at the call that leaks**, not rounds ahead of it.

<p align="center">
  <img src="assets/readme/fig6_trajectory_aggregate.png" width="85%" alt="Median probe score percentile aligned on the leaking call">
</p>

**4. Most of the gain comes from training on labels, not from activations.** A zero-shot text monitor does far worse than the probe. A bag-of-words classifier trained on the same labelled calls comes close: on the Qwen3-32B test episodes, the probe's edge is +0.02 AUROC, and its interval includes zero.

<p align="center">
  <img src="assets/readme/fig3_percall.png" width="85%" alt="Per-call AUROC: probe, zero-shot monitor and trained text classifier">
</p>

**5. The probe mostly reads a coming disclosure, not the rule.** It also flags permitted disclosures. Right after the target reads the secret, nearly every disclosure crosses the threshold, permitted or forbidden. Away from a read, forbidden leaks cross about twice as often as permitted shares (61% against 33%).

<p align="center">
  <img src="assets/readme/fig_e15_recency.png" width="70%" alt="Share of disclosures crossing the threshold by calls since the last read, forbidden versus permitted">
</p>

**6. Acting on the warning has a cost.** Flags cluster in the same episodes. At the per-call threshold, 59% of refusing Qwen3-32B episodes see at least one flag. A stricter episode-level threshold cuts false alarms but catches far fewer leaks, and a second text check does not close that gap.

<p align="center">
  <img src="assets/readme/fig7_qwen_alarms.png" width="60%" alt="Share of leaking calls caught against share of refusing-episode calls flagged">
</p>

**Scope.** Six scenarios built from one task template; one peer and one target per episode; synthetic tools and exact-string secrets; two open-weight models. See the paper's Limitations and Appendix N.

## What is in this repository

- **A recruitment benchmark:** six scenarios in which the same model leaks in some episodes and refuses in others, plus control episodes in which the peer never asks. All prompts and the Qwen3-32B transcripts are included.
- **Detectors scored on exactly the same calls:**
  - last-prompt and reasoning probes;
  - a zero-shot text monitor;
  - a trained bag-of-words classifier.
- **Every number in the paper, traced to its run.** Each run folder records its command, settings and outputs. [`paper-arxiv/numbers.yaml`](paper-arxiv/numbers.yaml) maps every number to its source file and key.

| Folder | Contents |
|---|---|
| [`paper-arxiv/`](paper-arxiv/) | The paper: LaTeX source, figures, tables and the number registry |
| [`experiments/E01-behaviour-pilot/`](experiments/E01-behaviour-pilot/) | The multi-agent harness, the six scenarios, peer tactics and outcome scorers |
| [`experiments/E02-activation-replay/`](experiments/E02-activation-replay/) | GPU replay of every target request through the open weights (on Modal), with fidelity checks |
| [`experiments/E03-target-screen/`](experiments/E03-target-screen/) | Second-target screens and the gpt-oss-20b collections |
| [`experiments/E05-probes/`](experiments/E05-probes/) | Probe training and evaluation, including the `arxiv_*.py` scripts behind the paper |
| [`experiments/E06-monitors/`](experiments/E06-monitors/) | Text monitors and their scoring |
| [`experiments/E10`–`E16`](experiments/) | Follow-up experiments: probe-then-monitor cascade, matched monitor comparison, and permitted versus forbidden disclosures |
| [`runs/`](runs/) | One folder per run, each with its command, settings and outputs |
| [`docs/`](docs/), [`explainers/`](explainers/) | Design notes and plain-language explainers of the methods |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | A dated journal of results, decisions and dead ends |

## Reproducing the paper

You can read every result without running anything. The paper's reproducibility index (Appendix Q) maps each figure and headline number to its run folder and script.

### Setup

What you need:
- Python 3.11 or later and [uv](https://docs.astral.sh/uv/);
- to build the paper, a TeX Live install with `latexmk` (the packages are listed in [`paper-arxiv/scripts/texlive_packages.txt`](paper-arxiv/scripts/texlive_packages.txt));
- only to rerun experiments (paid):
  - an [OpenRouter](https://openrouter.ai) API key, for the agents and monitors;
  - a [Modal](https://modal.com) account, for the GPU replay. The replay downloads the public weights: about 65 GB for Qwen3-32B and 28 GB for gpt-oss-20b.

```bash
git clone https://github.com/leonidas1712/precrime-agents
cd precrime-agents
uv sync

# probe and replay environments (exact pinned versions)
uv venv experiments/E05-probes/.venv --python 3.11
uv pip sync --python experiments/E05-probes/.venv/bin/python experiments/E05-probes/requirements-lock.txt
uv venv experiments/E02-activation-replay/.venv --python 3.11
uv pip sync --python experiments/E02-activation-replay/.venv/bin/python experiments/E02-activation-replay/requirements-lock.txt
```

For paid steps:
1. Put `OPENROUTER_API_KEY=…` in a git-ignored `.env` at the repository root.
2. Run `uv run modal setup` once.

### Build the paper (free, about a minute)

```bash
cd paper-arxiv
../.venv/bin/python scripts/build.py     # writes main.pdf; fails on any unresolved reference or citation
```

Every number in the text is a macro that `generate_numbers.py` generates from [`numbers.yaml`](paper-arxiv/numbers.yaml). Each entry records the run folder and key it was read from.

### Full run data

The checksum-pinned [data-v1 release](https://github.com/leonidas1712/collusion-research/releases/tag/data-v1) holds the saved transcripts, per-call scores and monitor caches: about 1.6 GB, expanding to about 8.5 GB. Restore them with:

```bash
scripts/fetch_run_data.sh
```

Not in git:
- saved activations and the arrays built from them (the GPU replay regenerates them);
- fitted probe models;
- the gpt-oss-20b transcripts (in the data release);
- API keys.

### What it costs to rerun

Every paid script takes a hard spending cap and records what it was billed in its run folder.
- **Rebuilding figures, tables and the paper:** free.
- **Re-scoring probes:** free on a laptop CPU, once the activation arrays exist.
- **GPU replay on one A100-80GB:** about USD 8.5 for Qwen3-32B and about USD 8 for gpt-oss-20b.
- **Episode collection and text monitors (OpenRouter):** a few dollars per experiment.

## Safety

All tools are mocked in memory, and every protected value is a randomly generated fake. No real credential, system or network resource is touched. The recruiting peer prompts are released in full; the paper's Appendix O explains why.

## Origin

This work began at the [Apart Research AI Collusion Research Sprint](https://apartresearch.com/sprints/projects/recruited-precrime-for-collusion-predicting-when-an-agent-gives-in-to-a-recruiting-peer-with-activation-probes-and-text-monitors-80jb) (October 2026).
- **The sprint submission** is kept unchanged at [leonidas1712/collusion-research](https://github.com/leonidas1712/collusion-research), tag `apart-submission`.
- **This repository** carries the full history, including the commits that fixed the test plans before their data were collected.
- **Appendix P** of the paper lists the numbers that changed since the sprint.

AI coding agents wrote most of the code and ran the experiments under the author's direction. The paper's LLM Usage Statement gives the full account.

## Citation

```bibtex
@misc{raghav2026precrime,
  title  = {Pre-Crime for Agents, at the Last Second: Probing When a Recruited Agent Gives In},
  author = {Raghav, Leonidas},
  year   = {2026},
  note   = {arXiv preprint},
  url    = {https://github.com/leonidas1712/precrime-agents}
}
```

## License

MIT for this project's code ([`LICENSE`](LICENSE)). The transcripts contain outputs of third-party models, and the replay uses open weights; both stay under their providers' terms.
