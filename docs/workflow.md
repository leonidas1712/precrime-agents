# How we work: runs, experiments, and keeping track

This is the working system for the repo. It is meant to make it fast to try many things, possibly in parallel, without losing track or confidence. It doesn't assume any particular method: probes and steering are today's plan, and the setup should work just as well if we pivot to a different idea next week.

Two goals shape everything:

1. **Speed.** Starting an experiment, sweeping it or running it in parallel takes minutes, with no framework to fight.
2. **Legibility.** At any point, and at the end, someone can open the repo and see what we tried, what we found, and how to reproduce each number. That serves the report and a public repo worth showing.

## The pieces

```
README.md                 what this project is, current findings, how to reproduce them
EXPERIMENT_LOG.md         chronological journal: experiments, decisions, pivots, dead ends
CLAIMS.md                 each claim we might report → experiment + run IDs + figure
docs/                     plan, infra, this file
src/collusion/            shared code, once something is used by more than one experiment
experiments/
  E01-short-name/         one folder per question we try to answer
    README.md             question, what we did, result, run IDs, status
    *.py                  the scripts for this experiment
  E02-.../
runs/                     one folder per run, created automatically (see below)
report/                   draft and report figures
```

### Experiments, not techniques

The repo is organised around **questions we tried to answer**, not around methods. A new idea, a paper to reproduce, a pivot or a fallback just becomes a new `experiments/ENN-name/` folder. Nothing existing has to be restructured.

- **Each experiment's README** stays short:
  - the question;
  - what we ran (with run IDs);
  - what we found;
  - a status: *active*, *done*, *inconclusive* or *abandoned (because …)*.
- **Abandoned and inconclusive experiments stay in the repo** with their reason. They show the path we took, and they stop anyone retrying a dead end.
- **Code starts inside the experiment that needs it.** Once a second experiment needs the same thing, move it into `src/collusion/`. The shared library then grows from real use instead of being designed up front for one method. Examples of things likely to end up there: model loading, the mock environment, activation capture, plotting.

### Runs record themselves

Every script that produces results starts with one call to a small run helper in `src/collusion/runs.py`. The helper creates `runs/<YYYYMMDD-HHMM>-<name>/` and writes:

- **`meta.json`:**
  - the settings actually used (plain function arguments or a dataclass; no config files needed);
  - the exact command;
  - the git commit, with any uncommitted changes saved as `diff.patch`. This includes staged, unstaged and new untracked source files, which a plain `git diff` misses. It means we don't have to commit before every run;
  - the model and its pinned version;
  - the machine, GPU and start time;
  - the IDs of any **earlier runs this one builds on**;
  - at exit, the status (finished or failed, with the error) and the end time.
- **`log.txt`:** everything the script printed.
- **Outputs as they are produced,** so a crash keeps whatever finished.

Run folder names get a short random suffix if needed, so runs started at the same minute never collide. The helper also appends one line to `runs/index.jsonl` when a run starts and when it ends: ID, experiment, status, tags, headline numbers. `python -m collusion.runs` prints the index as a table you can filter by experiment, tag, status or date; `--limit N` keeps only the last N rows. Blank or half-written lines in the index (from a run killed mid-write) are skipped. That table is the "what has been happening" view, and it works with many parallel jobs and no coordination.

**What gets committed:**
- `meta.json`, small results and figures, so readers of the public repo can trace every number.
- Large files (episodes at scale, activations, model weights) live on the persistent filesystem. Their run folder keeps a pointer to them.

### Lineage by run ID

Work naturally comes in stages: something generates data, something reads it, something analyses that, and something makes a figure. Each stage is a separate run that names the run IDs it read from. Any figure can then be traced back through every run that fed it, without copying data between folders. Re-running a later stage never touches the earlier one.

## Running lots of things

- **Sweeps** are a Python loop over settings, with each setting its own run. Tag them with a shared sweep name so the index groups them.
- **Many episodes or requests at once:** one model server per GPU, with many concurrent clients. Each unit of work writes its own output file, so nothing collides.
- **Restarting:** batch jobs skip any item whose output is complete. Outputs are written to a temporary name and renamed when finished, so a half-written file is never mistaken for a finished one. A crashed or interrupted job is restarted by running it again.
- **Sharing a GPU:** GPU jobs on the same card go through a simple queue, so two jobs don't run out of memory together. More GPUs or more machines on the same persistent storage simply means more workers writing their own run folders.
- **Exploring vs reporting:** notebooks are fine for exploring. Anything that produces a number or figure for the report becomes a script that takes run IDs and writes a run of its own.

## Staying confident without slowing down

A short list of habits, chosen because each prevents a likely error.

- **Separate choosing from reporting.** Anything picked by looking at results (a layer, a threshold, a prompt variant, a method) is chosen on one set of data, and the reported number comes from a different set. Split by episode or by whatever the independent unit is.
- **Always have a comparison.** Every result sits next to a control or baseline. When a method's output could also be explained by a simpler one (for example, reading the text), report the simpler baseline too.
- **Test the wiring separately from the science.** A fake model mode runs the whole pipeline without a GPU. Code tests check the code; a few checks on loaded results catch broken runs (missing or duplicated items, NaNs, the same unit in both splits).
- **Look at the raw thing.** Before trusting a number, read a few actual inputs and outputs: the literal prompt the model saw, a few labelled episodes.
- **Show that an intervention did something.** With the intervention on, outputs or activations should differ; with it off, they should match the baseline.
- **Keep every result,** including failed runs, sweeps and negative results. Never edit result files by hand: fix the code and re-run.
- **Numbers come from files.** A number in a board post, the log or the report must point to a run.

## Keeping the story readable

- **`EXPERIMENT_LOG.md`** is a chronological journal: short dated entries pointing to experiments and run IDs. Tag entries `[result]`, `[decision]`, `[pivot]`, `[dead-end]`, `[infra]` or `[idea]`. Decisions and pivots go here with their reason, so the history explains itself.
- **Board posts** lead with the result: the number, what it's on (n, which episodes), a caveat line, the run ID, and what's next.
- **Each experiment's README explains its setup in plain words,** written as we build it, not reconstructed at the end. For an agent setup: who acts when, who sees what, how a turn ends, which tools exist and what they do, what counts as an incomplete episode, and every condition we vary. A reader should be able to picture an episode without reading code. This section becomes the methods section of the report.
- **`CLAIMS.md`** lists what we might say in the report, each with its supporting runs and figure. The report is written from it.
- **The top-level `README.md`** is kept current: what the project is, the findings so far (linking experiments), and how to reproduce them. Someone landing on the repo cold should get the story from it in two minutes.
