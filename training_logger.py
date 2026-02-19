import json
import os
import time


class TrainingLogger:
    """JSONL format per-batch metrics logger with in-memory cache."""

    def __init__(self, output_dir):
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "aurora_metrics.jsonl")
        self._metrics = []

    def log_batch(self, metrics_dict):
        """Append one JSON line and keep an in-memory copy."""
        metrics_dict["_timestamp"] = time.time()
        self._metrics.append(metrics_dict)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics_dict, ensure_ascii=False) + "\n")

    def get_all_metrics(self):
        """Return all in-memory metrics (list of dicts)."""
        return list(self._metrics)

    @staticmethod
    def load_from_file(path):
        """Load metrics from a JSONL file (standalone usage)."""
        metrics = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
        return metrics
