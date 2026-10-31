"""Finite test runs: per-episode logs, weighted summaries and best-episode selection."""

from contextlib import ExitStack
import json
import math
from numbers import Integral
from pathlib import Path


def validate_test_episodes(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
        raise ValueError("test_episodes must be null or a positive integer")
    return int(value) if value is not None else None


def episode_selection_key(record):
    """Compare original precision; missing/nonfinite metrics rank below valid values."""
    def score(name, minimize=False):
        value = record.get(name)
        if value is None or not math.isfinite(value):
            return -math.inf
        return -value if minimize else value

    return (score("steps"), score("operation_completion_ratio"),
            score("planned_semantic_contact_coverage"), score("black_violation_frame_ratio", True))


class EpisodeStatistics:
    def __init__(self, limit, output_dir, print_stats=True):
        self.limit = validate_test_episodes(limit)
        if self.limit is None or output_dir is None:
            raise ValueError("Finite test statistics require test_episodes and an output directory")
        self.output_dir = Path(output_dir)
        self.print_stats = print_stats
        self.count = 0
        self.reward_sum = 0.0
        self.steps_sum = 0.0
        self.best = None
        self.semantic_enabled = None
        self.semantic_totals = None
        self.semantic_counts = {}
        self._files = ExitStack()

    @property
    def complete(self):
        return self.count == self.limit

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.log_file = self._files.enter_context((self.output_dir / "episodes.log").open("x", encoding="utf-8"))
            self.jsonl_file = self._files.enter_context((self.output_dir / "episodes.jsonl").open("x", encoding="utf-8"))
        except BaseException:
            self._files.close()
            raise
        return self

    def add_batch(self, done_ids, rewards, steps, snapshot=None):
        """Accept the first remaining ended environments, ordered by environment ID."""
        ids = done_ids.detach().cpu().flatten().tolist()
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Ended environment IDs must be nonempty and unique")
        if self.semantic_enabled is not None and self.semantic_enabled != (snapshot is not None):
            raise ValueError("Semantic evaluation availability changed during the test run")

        rows = {}
        if snapshot is not None:
            snapshot = {name: value.detach().cpu() for name, value in snapshot.items()}
            snapshot_ids = snapshot["env_ids"].tolist()
            if sorted(snapshot_ids) != sorted(ids):
                raise ValueError("Semantic terminal snapshot must match exactly the ended environments")
            if any(value.ndim != 1 or len(value) != len(ids) for value in snapshot.values()):
                raise ValueError("Semantic terminal snapshot fields must have one value per ended environment")
            rows = {env_id: row for row, env_id in enumerate(snapshot_ids)}
            if self.semantic_totals is None:
                from main.semantic_contact_eval import SemanticContactSummary
                self.semantic_totals = SemanticContactSummary()

        accepted = sorted(ids)[:self.limit - self.count]
        for env_id in accepted:
            reward, length = float(rewards[env_id].item()), float(steps[env_id].item())
            if not math.isfinite(reward) or not math.isfinite(length):
                raise ValueError("Episode reward and steps must be finite")
            record = {"episode": self.count + 1, "env_id": env_id, "reward": reward, "steps": length}
            message = (f"episode: {record['episode']}/{self.limit} env_id: {env_id} "
                       f"reward: {reward:.2f} steps: {length:.1f}")
            if snapshot is not None:
                from main.semantic_contact_eval import format_semantic_contact_summary
                row = rows[env_id]
                single = {name: value[row:row + 1] for name, value in snapshot.items()}
                summary = self.semantic_totals.add(single, single["env_ids"])
                record.update(summary)
                record["semantic_counts"] = {name: value.item() for name, value in single.items() if name != "env_ids"}
                message += " " + format_semantic_contact_summary(summary)
                for name, value in record["semantic_counts"].items():
                    self.semantic_counts[name] = self.semantic_counts.get(name, 0) + int(value)
                self.semantic_counts["no_contact_episodes"] = (
                    self.semantic_counts.get("no_contact_episodes", 0) + summary["no_contact_episodes"])
            record["log"] = message
            self.jsonl_file.write(json.dumps(record, allow_nan=False) + "\n")
            self.jsonl_file.flush()
            self.log_file.write(message + "\n")
            self.log_file.flush()
            self.semantic_enabled = snapshot is not None
            self.count += 1
            self.reward_sum += reward
            self.steps_sum += length
            if self.best is None or episode_selection_key(record) > episode_selection_key(self.best):
                self.best = record
            if self.print_stats:
                print(message, flush=True)
        return accepted

    def semantic_summary(self):
        if not self.semantic_counts:
            return None
        c = self.semantic_counts
        events = c["red_events"] + c["black_events"]
        return {
            "operation_completion_ratio": c["eval_frames"] / c["planned_frames"],
            "planned_semantic_contact_coverage": c["compliant_required_frames"] / c["planned_required_frames"]
            if c["planned_required_frames"] else None,
            "black_violation_frame_ratio": c["black_frames"] / c["eval_frames"] if c["eval_frames"] else None,
            "semantic_task_success_rate": c["semantic_success"] / self.count,
            "semantic_sr_total": c["semantic_success"] / self.count,
            "red_contact_ratio": c["red_events"] / events if events else None,
            "black_contact_ratio": c["black_events"] / events if events else None,
            "eval_frames": c["eval_frames"], "planned_frames": c["planned_frames"],
            "required_frames": c["planned_required_frames"],
            "compliant_required_frames": c["compliant_required_frames"],
            "black_required_frames": c["black_required_frames"],
            "missing_contact_frames": c["missing_contact_frames"],
            "unexecuted_required_frames": c["unexecuted_required_frames"],
            "complete_episodes": c["complete"], "episodes": self.count,
            "no_contact_episodes": c["no_contact_episodes"],
        }

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            summary = {
                "target_episodes": self.limit, "completed_episodes": self.count,
                "status": "complete" if self.complete and exc_type is None else "incomplete",
                "stop_reason": exc_type.__name__ if exc_type is not None else
                ("episode_limit" if self.complete else "early_exit"),
                "mean_reward": self.reward_sum / self.count if self.count else None,
                "mean_steps": self.steps_sum / self.count if self.count else None,
                "max_steps": self.best["steps"] if self.best is not None else None,
                "selection_order": ["steps:desc", "operation_completion_ratio:desc",
                                    "planned_semantic_contact_coverage:desc", "black_violation_frame_ratio:asc"],
                "best_episode": self.best, "semantic_contact": self.semantic_summary(),
            }
            with (self.output_dir / "summary.json").open("w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2, allow_nan=False)
                file.write("\n")
            if self.print_stats:
                print(f"[test_summary] completed_episodes: {self.count}/{self.limit} "
                      f"status: {summary['status']} mean_reward: {summary['mean_reward']} "
                      f"mean_steps: {summary['mean_steps']} max_steps: {summary['max_steps']}", flush=True)
                if summary["semantic_contact"] is not None:
                    from main.semantic_contact_eval import format_semantic_contact_summary
                    print("[test_semantic_summary] " + format_semantic_contact_summary(summary["semantic_contact"]), flush=True)
                if self.best is not None:
                    print("[best_steps] " + self.best["log"], flush=True)
        finally:
            self._files.close()
