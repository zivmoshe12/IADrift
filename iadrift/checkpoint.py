"""Resumable experiment checkpoints.

An experiment is described by a spec: the run plan (one dict per optimization run) and all settings. The first 12 hex
digits of the spec's SHA-256 go into the file names, so results from a changed setting never mix with old ones:

    <stem>.manifest.json   the spec; written once and checked on every resume
    <stem>.jsonl           one JSON row per finished run, rewritten atomically after each run
    <stem>_arrays/         one .npz per run (eps, outputs, loss traces)
    <stem>.lock            flock held while a process writes the checkpoint

Runners do at most --max-runs runs per call and print "ALL RUNS COMPLETE" when the plan is done; the overnight driver
(scripts/test/run_overnight.sh) stops on that line.
"""

import fcntl
import hashlib
import io
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- atomic files
def atomic_write_bytes(path, data):
    """Write to a temporary file, then atomically replace path; fsyncs the file and the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_save_results(path, rows):
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    atomic_write_bytes(path, payload.encode("utf-8"))


def load_results(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_npz(path, arrays, metrics=None):
    """Atomic compressed .npz. metrics (a dict) is stored as a JSON string under 'metrics_json'."""
    buffer = io.BytesIO()
    extra = {} if metrics is None else {"metrics_json": json.dumps(metrics)}
    np.savez_compressed(buffer, **extra, **arrays)
    atomic_write_bytes(path, buffer.getvalue())


def load_npz(path):
    """Inverse of save_npz: (metrics or None, {name: array})."""
    with np.load(path) as data:
        metrics = json.loads(str(data["metrics_json"])) if "metrics_json" in data.files else None
        return metrics, {k: data[k] for k in data.files if k != "metrics_json"}


# ---------------------------------------------------------------- hashed checkpoints
def spec_hash(spec):
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class Checkpoint:
    """JSONL checkpoint named by the spec hash: stem = <directory>/<prefix>_<hash>."""

    def __init__(self, directory, prefix, spec, key_fn):
        self.spec = spec
        self.spec_text, self.hash = spec_hash(spec)
        self.stem = Path(directory) / f"{prefix}_{self.hash}"
        self.results_path = Path(f"{self.stem}.jsonl")
        self.manifest_path = Path(f"{self.stem}.manifest.json")
        self.arrays_dir = Path(f"{self.stem}_arrays")
        self.key_fn = key_fn
        self.plan = spec["run_plan"]
        self.plan_by_key = {key_fn(run): run for run in self.plan}
        assert len(self.plan_by_key) == len(self.plan), "run plan has duplicate keys"

    def n_done(self):
        return len(load_results(self.results_path)) if self.results_path.exists() else 0

    def check_manifest(self):
        if self.manifest_path.exists() and json.loads(self.manifest_path.read_text(encoding="utf-8")) != self.spec:
            raise RuntimeError(f"Existing manifest does not match this plan: {self.manifest_path}")

    @contextmanager
    def open(self):
        """Take the lock, write or check the manifest, and yield (results so far, runs still to do)."""
        with Path(f"{self.stem}.lock").open("a+") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Checkpoint is locked by another process.") from error
            if self.manifest_path.exists():
                self.check_manifest()
            else:
                if self.results_path.exists():
                    raise RuntimeError("Results exist without a manifest; refusing to mix runs.")
                atomic_write_bytes(self.manifest_path, (self.spec_text + "\n").encode("utf-8"))
            results = load_results(self.results_path) if self.results_path.exists() else []
            completed = {self.key_fn(row) for row in results}
            if len(completed) != len(results) or not completed <= self.plan_by_key.keys():
                raise RuntimeError("Checkpoint has duplicate or unexpected runs.")
            yield results, [run for run in self.plan if self.key_fn(run) not in completed]

    def append(self, results, row):
        atomic_save_results(self.results_path, results + [row])
        results.append(row)

    def finish(self, results):
        print(f"Saved {len(results)} / {len(self.plan)} results to {self.results_path}")
        if len(results) == len(self.plan):
            print(f"ALL RUNS COMPLETE: {self.results_path}")


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9
