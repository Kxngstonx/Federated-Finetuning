"""Result logging: one JSONL file per run (metrics.jsonl, one line per round/phase event) plus a
run_metadata.json written once at run start (GPU info, seed, task, strategy, lr). Both live
alongside the existing results/<timestamp>/ checkpoint dirs that server_app.py::server_fn
already creates in both flowertune_llm and the two new apps.

This supports the required post-hoc aggregation: mean +/- std over 3 seeds (aggregate_glue_
results.py / aggregate_nlg_results.py group runs by run_metadata.json's (task, strategy, lr) and
join to each run's metrics.jsonl), and per-round/per-checkpoint learning-curve plotting (each
metrics.jsonl is directly readable via `pandas.read_json(path, lines=True)`).
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class MetricsWriter:
    path: Path  # e.g. results/<timestamp>/metrics.jsonl

    def write_round(self, round: int, phase: str, **fields: Any) -> None:
        record = {"round": round, "phase": phase, "timestamp": time.time(), **fields}
        with open(self.path, "a") as f:
            # default=str: tolerate stray numpy scalars/arrays in `fields` without crashing a run
            f.write(json.dumps(record, default=str) + "\n")


def write_run_metadata(save_path: Path, metadata: Dict[str, Any]) -> None:
    """Write once at server_fn startup: gpu_type, gpu_count, seed, task/dataset, strategy, lr,
    start_time. Read back by aggregate_*_results.py to group runs for mean +/- std over seeds."""
    with open(Path(save_path) / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
