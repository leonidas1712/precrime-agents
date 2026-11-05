"""E17 activation replay: last-prompt-token residuals of Qwen3-32B for the 423 scored calls.

Copy of experiments/E15-giving-in-scaleup/modal_app.py with the frozen-probe scoring and reply teacher-forcing removed.
Importing this module does not submit work.

    .venv/bin/modal run experiments/E17-crossed/replay_modal.py --mode benchmark --payload-path private/crossed/e17_replay.payload.json
    .venv/bin/modal run experiments/E17-crossed/replay_modal.py --mode full --payload-path private/crossed/e17_replay.payload.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/app/e17")
ROOT = HERE.parents[1] if modal.is_local() else Path("/app")
OLD_ROOT = Path("/Users/leo/collusion-research")
E02 = ROOT / "experiments/E02-activation-replay" if modal.is_local() else Path("/app/e02")
E08 = ROOT / "experiments/E08-policy-counterfactual" if modal.is_local() else Path("/app/e08")
sys.path[:0] = [str(HERE), str(E02), str(E08)]

app = modal.App("collusion-e17-crossed-replay")
weights_volume = modal.Volume.from_name("collusion-weights", create_if_missing=False)
outputs_volume = modal.Volume.from_name("collusion-activations", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.4.6",
        "huggingface-hub==2.3.0",
    )
    .env({"HF_HOME": "/weights/huggingface", "PYTHONPATH": "/app/src:/app/e02:/app/e08"})
)
if modal.is_local():
    image = image.add_local_dir(ROOT / "src" / "collusion", remote_path="/app/src/collusion", ignore=["__pycache__/**"])
    image = image.add_local_file(E02 / "replay.py", "/app/e02/replay.py")
    image = image.add_local_file(E02 / "fidelity.py", "/app/e02/fidelity.py")
    image = image.add_local_file(E08 / "counterfactual.py", "/app/e08/counterfactual.py")


def execute(plan: dict, payload: list[dict], provenance: dict, mode: str) -> dict:
    import csv
    import os

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from counterfactual import digest
    from replay import capture, decoder_layers, render

    if mode not in ("benchmark", "full"):
        raise ValueError("mode must be benchmark or full")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_spec = plan["model"]
    model_id = model_spec["name"]
    revision = model_spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pinned 40-character model revision required")
    marker = Path("/weights/ready") / model_id.replace("/", "--") / f"{revision}.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Prefetched model marker missing: {marker}")
    if mode == "benchmark":
        payload = payload[: plan["gpu"]["benchmark_calls"]]

    import collusion.runs as run_helpers

    run_helpers.RUNS = Path("/outputs/runs")
    run_helpers.INDEX = run_helpers.RUNS / "index.jsonl"
    run = run_helpers.start_run(
        "e17-crossed-replay",
        experiment="E17",
        settings={"mode": mode, "plan_sha256": provenance["plan_sha256"]},
        parents=provenance["parents"],
        model=model_id,
        tags=["e17", "prefill", mode],
    )
    wall_started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False, local_files_only=True)
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False, local_files_only=True)

        # Validate every edit and render before allocating model weights.
        rendered_inputs = []
        for item in payload:
            request = item["request"]
            rendered, ids, changes, thinking = render(tokenizer, request, "from-request")
            if thinking:
                raise ValueError(f"Reasoning was off in the run but the render enabled it: {item['call']['call_id']}")
            if len(ids) > config.max_position_embeddings:
                raise ValueError(f"Prompt exceeds context: {item['call']['call_id']}")
            rendered_inputs.append((item, request, rendered, ids, changes))

        load_started = time.perf_counter()
        causal_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model = causal_model.base_model
        model.eval()
        model_load_seconds = time.perf_counter() - load_started
        layers_path, layers = decoder_layers(model)
        selected_layers = [int(layer) for layer in model_spec["layers"]]
        capture_layers = [layers[layer] for layer in selected_layers]
        input_device = model.get_input_embeddings().weight.device
        torch.cuda.reset_peak_memory_stats()

        rows = []
        activation_seconds = 0.0
        prompt_tokens = 0
        forward_tokens = 0
        activation_dir = run.dir / "activations"
        activation_dir.mkdir(parents=True, exist_ok=True)
        for index, (item, request, rendered, ids, changes) in enumerate(rendered_inputs):
            if mode == "full" and time.perf_counter() - wall_started > plan["gpu"]["full_timeout_seconds"]:
                raise TimeoutError("E17 full-run wall guard reached")
            started = time.perf_counter()
            values = capture(model, capture_layers, ids, input_device, prompt_length=len(ids))
            elapsed = time.perf_counter() - started
            activation_seconds += elapsed
            prompt_tokens += len(ids)
            forward_tokens += len(ids)
            key = hashlib.sha256(item["call"]["call_id"].encode()).hexdigest()[:20]
            with (activation_dir / f"{key}.tmp").open("wb") as handle:
                np.savez_compressed(
                    handle,
                    residual_last=values.astype(np.float32),
                    layer_indices=np.asarray(selected_layers, dtype=np.int64),
                    input_ids=np.asarray(ids, dtype=np.int64),
                )
            (activation_dir / f"{key}.tmp").replace(activation_dir / f"{key}.npz")
            rows.append({
                **{k: v for k, v in item["call"].items() if k != "setup"},
                "n_tokens": len(ids),
                "input_ids_sha256": digest(ids),
                "rendered_prompt_sha256": digest(rendered.encode()),
                "edited_request_sha256": digest(request),
                "normalization_count": len(changes),
                "activation_seconds": elapsed,
                "activation_file": f"activations/{key}.npz",
            })
            if (index + 1) % 25 == 0:
                print(json.dumps({"completed": index + 1, "total": len(rendered_inputs), "prompt_tokens": prompt_tokens}), flush=True)
                outputs_volume.commit()

        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (run.dir / "calls.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "mode": mode,
            "source_calls": len(payload),
            "forwards": len(rows),
            "prompt_tokens": prompt_tokens,
            "forward_tokens": forward_tokens,
            "model_load_seconds": model_load_seconds,
            "activation_seconds": activation_seconds,
            "wall_seconds": time.perf_counter() - wall_started,
            "tokens_per_activation_second": prompt_tokens / activation_seconds,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "selected_layers": selected_layers,
            "layers_path": layers_path,
            "plan_sha256": provenance["plan_sha256"],
        }
        (run.dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        (run.dir / "input_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        (run.dir / "gpu_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        run.result(**summary)
        run.finish()
        outputs_volume.commit()
        return {"path": str(run.dir.relative_to("/outputs")), "summary": summary}
    except BaseException as error:
        run.finish("failed", f"{type(error).__name__}: {error}")
        outputs_volume.commit()
        raise


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=20 * 60,
    volumes={"/weights": weights_volume, "/outputs": outputs_volume},
    max_containers=1,
    cpu=(4, 4),
    memory=(32768, 65536),
    scaledown_window=2,
)
def benchmark_remote(plan: dict, payload: list[dict], provenance: dict):
    return execute(plan, payload, provenance, "benchmark")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=55 * 60,
    volumes={"/weights": weights_volume, "/outputs": outputs_volume},
    max_containers=1,
    cpu=(4, 4),
    memory=(32768, 65536),
    scaledown_window=2,
)
def full_remote(plan: dict, payload: list[dict], provenance: dict):
    return execute(plan, payload, provenance, "full")


@app.local_entrypoint()
def main(mode: str = "benchmark", payload_path: str = ""):
    if mode not in ("benchmark", "full"):
        raise ValueError("mode must be benchmark or full")
    from counterfactual import digest

    path = HERE / "replay_plan.json"
    plan = json.loads(path.read_text())
    rel = str(path.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode or dirty.stdout.strip():
        raise ValueError("replay_plan.json must be committed and clean before a GPU call")
    payload_file = Path(payload_path)
    payload = json.loads(payload_file.read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    provenance = {
        "plan_sha256": digest(path.read_bytes()),
        "payload_sha256": digest(payload_file.read_bytes()),
        "local_git_commit": commit,
        "parents": sorted({item["call"]["episode"].split("/", 1)[0] for item in payload}),
        "code_sha256": {name: digest((HERE / name).read_bytes()) for name in ("replay_modal.py", "replay_payload.py")},
    }
    function = benchmark_remote if mode == "benchmark" else full_remote
    result = function.remote(plan, payload, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("mkdir -p ./modal-results")
    print(f'modal volume get collusion-activations {result["path"]} ./modal-results')
