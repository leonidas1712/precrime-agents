import json

from collusion import runs


def _write_index(path, entries, extra=""):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries) + extra)


def test_load_index_merges_start_and_end(tmp_path, monkeypatch):
    index = tmp_path / "index.jsonl"
    _write_index(index, [
        {"id": "a", "event": "start", "experiment": "E01", "status": "running", "tags": [], "time": "t0"},
        {"id": "a", "event": "end", "experiment": "E01", "status": "finished", "tags": [], "results": {"acc": 0.4}, "time": "t1"},
    ])
    monkeypatch.setattr(runs, "INDEX", index)
    [row] = runs.load_index()
    assert row["status"] == "finished"
    assert row["results"] == {"acc": 0.4}


def test_load_index_skips_blank_and_truncated_lines(tmp_path, monkeypatch):
    index = tmp_path / "index.jsonl"
    _write_index(index, [
        {"id": "a", "event": "start", "experiment": "E01", "status": "running", "tags": [], "time": "t0"},
    ], extra='\n{"id": "b", "event": "sta')
    monkeypatch.setattr(runs, "INDEX", index)
    assert [r["id"] for r in runs.load_index()] == ["a"]


def test_limit_keeps_last_rows(tmp_path, monkeypatch, capsys):
    index = tmp_path / "index.jsonl"
    _write_index(index, [
        {"id": i, "event": "start", "experiment": "E01", "status": "running", "tags": [], "time": "t"} for i in "abc"
    ])
    monkeypatch.setattr(runs, "INDEX", index)
    runs.main(["--limit", "2"])
    printed = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert printed == ["b", "c"]
