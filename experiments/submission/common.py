"""Receipts shared by the submission experiments (no import-time execution)."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(problem):
    value = [sorted(problem["numbers"]), problem["target"]]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def environment():
    import torch
    def output(command):
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    versions={}
    for name in ("torch","numpy","transformers","safetensors","huggingface_hub","matplotlib","pytest"):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return {"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"hostname":platform.node(),
            "libraries":versions,"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "git_head": output(["git", "rev-parse", "HEAD"]),
            "git_status": output(["git", "status", "--short"]),
            "hardware": output(["lscpu"]),
            "gpu": output(["nvidia-smi"]), "argv": sys.argv,
            "source_hashes": {str(p.relative_to(ROOT)): digest(p)
                              for p in sorted((ROOT / "experiments/submission").glob("*.py"))},
            "domain_hash": digest(ROOT / "interference_search/countdown.py"),
            "judge_source_hash": digest(ROOT / "interference_search/judge.py")}


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
