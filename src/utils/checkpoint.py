import json
import os
from loguru import logger


class CheckpointManager:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {len(data.get('completed_runs', []))} completed runs, "
                        f"{len(data.get('phase1_results', []))} phase1 results")
            return data
        return {"completed_runs": [], "phase1_results": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.checkpoint_path) or '.', exist_ok=True)
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_completed(self, run_tag):
        return run_tag in self.data["completed_runs"]

    def mark_completed(self, run_tag):
        if run_tag not in self.data["completed_runs"]:
            self.data["completed_runs"].append(run_tag)
            self._save()

    def add_phase1_result(self, result):
        self.data["phase1_results"].append(result)
        self._save()

    def get_phase1_results(self):
        return list(self.data.get("phase1_results", []))

    def is_phase_complete(self, phase):
        return self.data.get(f"phase_{phase}_complete", False)

    def mark_phase_complete(self, phase):
        self.data[f"phase_{phase}_complete"] = True
        self._save()
