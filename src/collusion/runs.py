"""Run helper: every script that produces results starts with `start_run(...)`.

Creates runs/<YYYYMMDD-HHMM>-<name>/ with meta.json, log.txt (everything printed) and
diff.patch (uncommitted changes, including untracked files), and appends start/end lines
to runs/index.jsonl. See docs/workflow.md.

    run = start_run("pilot-cred", experiment="E01", settings={...}, parents=["20261023-2310-gen"])
    run.dir / "episodes"      # write outputs here
    run.result(acceptance=0.4) # headline numbers, saved at exit

`python -m collusion.runs [--exp E01] [--status finished] [--tag sweep1] [--limit 20]` prints the index.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import secrets
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
INDEX = RUNS / "index.jsonl"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _diff_patch() -> str:
    """Staged + unstaged changes vs HEAD, plus untracked files (which plain `git diff` misses)."""
    patch = _git("diff", "HEAD")
    for path in _git("ls-files", "--others", "--exclude-standard").splitlines():
        if path.startswith("runs/"):
            continue
        out = subprocess.run(["git", "diff", "--no-index", "/dev/null", path], cwd=REPO, capture_output=True, text=True)
        patch += out.stdout
    return patch


def _gpu() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except FileNotFoundError:
        return None


def write_json_atomic(path: Path, obj) -> None:
    """Write to a temp name then rename, so a half-written file is never taken for a finished one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _append_index(entry: dict) -> None:
    RUNS.mkdir(exist_ok=True)
    with open(INDEX, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        self.log.write(s)
        self.log.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@dataclass
class Run:
    id: str
    dir: Path
    meta: dict
    results: dict = field(default_factory=dict)

    def result(self, **numbers) -> None:
        """Record headline numbers; saved to meta.json and the index at exit."""
        self.results.update(numbers)

    def finish(self, status: str = "finished", error: str | None = None) -> None:
        """Mark the run finished or failed. Only the first call counts; at exit, an unfinished run is
        marked failed if an exception escaped, else finished. Call finish("failed", ...) yourself when
        failures are caught (e.g. some episodes errored) or on a nonzero sys.exit."""
        if self.meta.get("status") != "running":
            return
        self.meta.update(status=status, error=error, ended=datetime.now().isoformat(timespec="seconds"), results=self.results)
        write_json_atomic(self.dir / "meta.json", self.meta)
        _append_index({"id": self.id, "event": "end", "experiment": self.meta["experiment"], "status": status,
                       "tags": self.meta["tags"], "results": self.results, "time": self.meta["ended"]})


def start_run(name: str, experiment: str, settings: dict | None = None, parents: list[str] | None = None,
              tags: list[str] | None = None, model: str | None = None) -> Run:
    """Create the run folder, record what is being run, and tee stdout/stderr into log.txt."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = RUNS / f"{stamp}-{name}"
    while True:
        try:
            run_dir.mkdir(parents=True)
            break
        except FileExistsError:  # another run started in the same minute
            run_dir = RUNS / f"{stamp}-{name}-{secrets.token_hex(2)}"
    run_id = run_dir.name

    patch = _diff_patch()
    if patch:
        (run_dir / "diff.patch").write_text(patch)
    meta = {
        "id": run_id,
        "experiment": experiment,
        "name": name,
        "tags": tags or [],
        "settings": settings or {},
        "parents": parents or [],
        "model": model,
        "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        "git_commit": _git("rev-parse", "HEAD").strip() or None,
        "dirty": bool(patch),
        "host": platform.node(),
        "gpu": _gpu(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }
    write_json_atomic(run_dir / "meta.json", meta)
    _append_index({"id": run_id, "event": "start", "experiment": experiment, "status": "running",
                   "tags": meta["tags"], "time": meta["started"]})

    log = open(run_dir / "log.txt", "a")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
    run = Run(run_id, run_dir, meta)

    state = {"error": None}
    prev_hook = sys.excepthook

    def hook(exc_type, exc, tb):
        state["error"] = "".join(traceback.format_exception(exc_type, exc, tb))
        prev_hook(exc_type, exc, tb)

    sys.excepthook = hook

    def at_exit():
        run.finish("failed" if state["error"] else "finished", state["error"])

    atexit.register(at_exit)
    display_dir = run_dir.relative_to(REPO) if run_dir.is_relative_to(REPO) else run_dir
    print(f"[run] {run_id} -> {display_dir}")
    return run


def load_index() -> list[dict]:
    """One row per run: the start entry merged with its end entry, if any."""
    if not INDEX.exists():
        return []
    rows: dict[str, dict] = {}
    for line in INDEX.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:  # a line cut short by a crash mid-write
            continue
        row = rows.setdefault(e["id"], {"id": e["id"], "experiment": e["experiment"], "tags": e.get("tags", [])})
        if e["event"] == "start":
            row.update(started=e["time"], status="running")
        else:
            row.update(status=e["status"], ended=e["time"], results=e.get("results", {}))
    return list(rows.values())


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="List runs from runs/index.jsonl")
    p.add_argument("--exp")
    p.add_argument("--status")
    p.add_argument("--tag")
    p.add_argument("--since", help="YYYYMMDD")
    p.add_argument("--limit", type=int, help="show only the last N matching runs")
    a = p.parse_args(argv)
    rows = load_index()
    if a.exp:
        rows = [r for r in rows if r["experiment"] == a.exp]
    if a.status:
        rows = [r for r in rows if r["status"] == a.status]
    if a.tag:
        rows = [r for r in rows if a.tag in r["tags"]]
    if a.since:
        rows = [r for r in rows if r["id"][:8] >= a.since]
    if a.limit:
        rows = rows[-a.limit:]
    for r in rows:
        res = " ".join(f"{k}={v}" for k, v in (r.get("results") or {}).items())
        print(f"{r['id']:40s} {r['experiment']:6s} {r['status']:9s} {','.join(r['tags']):15s} {res}")


if __name__ == "__main__":
    main()
