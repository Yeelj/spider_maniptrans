"""CPU integration tests for finite player runs; no Isaac Gym or GPU required."""

import ast
from contextlib import redirect_stdout
import json
from io import StringIO
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

from lib.utils.episode_statistics import EpisodeStatistics, episode_selection_key, validate_test_episodes
from lib.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields


ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, name, extra=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "np": np, "stack_sequence_fields": stack_sequence_fields,
                 "unstack_sequence_fields": unstack_sequence_fields}
    namespace.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec"), namespace)
    return namespace[name]


def snapshot(env_ids, overrides=None):
    rows = []
    for env_id in env_ids:
        row = dict(red_events=2, black_events=0, eval_frames=2, black_frames=0, contact_frames=2,
                   planned_frames=2, planned_required_frames=2, executed_required_frames=2,
                   compliant_required_frames=2, black_required_frames=0, missing_contact_frames=0,
                   unexecuted_required_frames=0, task_success=True, complete=True, semantic_success=True)
        row.update((overrides or {}).get(env_id, {}))
        rows.append(row)
    result = {name: torch.tensor([row[name] for row in rows]) for name in rows[0]}
    result["env_ids"] = torch.tensor(env_ids)
    return result


def player(directory, limit, num_envs=4, semantics=True, episode_length=1, save_rollouts=False):
    env = SimpleNamespace(num_envs=num_envs, success_buf=torch.ones(num_envs, dtype=torch.bool),
                          dump_fileds={"state": torch.zeros(num_envs, 1)})
    calls = {"steps": 0, "resets": 0}

    def reset_done():
        calls["resets"] += 1
        return torch.zeros(num_envs, 1), None

    def step(env, actions):
        calls["steps"] += 1
        env.dump_fileds = {"state": torch.full((num_envs, 1), float(calls["steps"]))}
        done = torch.full((num_envs,), calls["steps"] % episode_length == 0, dtype=torch.bool)
        info = {}
        if semantics and done.any():
            # Snapshot row order must never be assumed to equal environment ID.
            info["semantic_contact_eval"] = snapshot(list(reversed(range(num_envs))))
        return torch.zeros(num_envs, 1), torch.arange(1., num_envs + 1), done, info

    env.reset_done = reset_done
    p = SimpleNamespace(test_episodes=limit, test_output_dir=directory, print_stats=True,
                        render_env=False, is_deterministic=True, save_rollouts=save_rollouts,
                        is_rnn=False, env=env, wait_for_checkpoint=lambda: None,
                        env_reset=lambda env: torch.zeros(num_envs, 1),
                        get_batch_size=lambda obs, batch: num_envs,
                        get_action=lambda obs, det: torch.zeros(num_envs, 1), env_step=step,
                        h5py_file=None)
    for name in ("run", "_run", "_save_completed_rollouts"):
        setattr(p, name, MethodType(method("lib/rl/player.py", "MyBasePlayer", name), p))
    if save_rollouts:
        p.h5py_file = h5py.File(str(Path(directory) / "rollouts.hdf5"), "w")
        p.s_grp = p.h5py_file.create_group("successful")
        p.f_grp = p.h5py_file.create_group("failed")
        p.saved_successful_rollouts = p.saved_failed_rollouts = p.prev_done_count_sum = 0
        p.min_episode_length = 2
        p.num_rollouts_to_save = 100
        p.num_rollouts_to_run = 1  # Must not override an explicit finite episode limit.
        p.save_successful_rollouts_only = True
    return p, calls


def read_results(directory):
    directory = Path(directory)
    records = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    return records, json.loads((directory / "summary.json").read_text())


class EpisodeStatisticsTests(unittest.TestCase):
    def test_limit_validation(self):
        self.assertIsNone(validate_test_episodes(None))
        for value in (1, 25, 100, np.int64(3)):
            self.assertEqual(validate_test_episodes(value), value)
        for value in (0, -1, True, False, 1.5, 100., "100", "", float("nan")):
            with self.assertRaises(ValueError):
                validate_test_episodes(value)

    def test_selection_priorities_and_original_precision(self):
        def key(steps=10, completion=.5, coverage=.5, violation=.1):
            return episode_selection_key(dict(steps=steps, operation_completion_ratio=completion,
                                              planned_semantic_contact_coverage=coverage,
                                              black_violation_frame_ratio=violation))
        self.assertGreater(key(steps=11, completion=0, coverage=0, violation=1), key())
        self.assertGreater(key(completion=.6, coverage=0, violation=1), key())
        self.assertGreater(key(coverage=.6, violation=1), key())
        self.assertGreater(key(violation=0), key())
        self.assertGreater(key(completion=.500001), key(completion=.500000))
        for value in (None, float("nan"), float("inf")):
            self.assertGreater(key(completion=0), key(completion=value))
            self.assertGreater(key(violation=1), key(violation=value))

    def test_best_semantic_record_uses_every_tiebreak_and_retains_first_exact_tie(self):
        variants = [dict(eval_frames=5, planned_frames=10),
                    dict(eval_frames=6, planned_frames=10, compliant_required_frames=1, planned_required_frames=10),
                    dict(eval_frames=6, planned_frames=10, compliant_required_frames=2, planned_required_frames=10,
                         black_frames=2),
                    dict(eval_frames=6, planned_frames=10, compliant_required_frames=2, planned_required_frames=10,
                         black_frames=1)]
        with tempfile.TemporaryDirectory() as directory:
            with EpisodeStatistics(5, directory, False) as stats:
                for i, values in enumerate(variants + [variants[-1]]):
                    stats.add_batch(torch.tensor([0]), torch.tensor([float(i)]), torch.tensor([10.]),
                                    snapshot([0], {0: values}))
                    self.assertEqual(stats.best["episode"], min(i + 1, 4))
            records, summary = read_results(directory)
            self.assertEqual(summary["best_episode"], records[3])
            self.assertEqual(summary["best_episode"]["semantic_sr_total"], 1)

    def test_weighted_summary_matches_combined_raw_counts(self):
        values = {0: dict(red_events=10, eval_frames=10, planned_frames=10, planned_required_frames=10,
                          compliant_required_frames=10),
                  1: dict(red_events=0, black_events=30, black_frames=10, eval_frames=20,
                          planned_frames=100, planned_required_frames=100, compliant_required_frames=10,
                          semantic_success=False, complete=False)}
        with tempfile.TemporaryDirectory() as directory:
            with EpisodeStatistics(2, directory, False) as stats:
                stats.add_batch(torch.tensor([1, 0]), torch.ones(2), torch.ones(2), snapshot([1, 0], values))
            records, summary = read_results(directory)
            self.assertEqual([r["env_id"] for r in records], [0, 1])
            combined = summary["semantic_contact"]
            self.assertAlmostEqual(combined["operation_completion_ratio"], 30 / 110)
            self.assertAlmostEqual(combined["planned_semantic_contact_coverage"], 20 / 110)
            self.assertAlmostEqual(combined["black_violation_frame_ratio"], 10 / 30)
            self.assertAlmostEqual(combined["red_contact_ratio"], 10 / 40)
            self.assertEqual(combined["semantic_sr_total"], .5)

    def test_missing_metrics_and_identical_steps_keep_first(self):
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            with redirect_stdout(output), EpisodeStatistics(2, directory) as stats:
                stats.add_batch(torch.tensor([0, 1]), torch.tensor([1., 999.]), torch.tensor([10., 10.]))
                # The log is already readable before leaving the context.
                self.assertEqual(len((Path(directory) / "episodes.log").read_text().splitlines()), 2)
            records, summary = read_results(directory)
            self.assertEqual(summary["best_episode"], records[0])
            self.assertIsNone(summary["semantic_contact"])
            self.assertEqual(output.getvalue().splitlines()[-1], "[best_steps] " + records[0]["log"])

    def test_bad_snapshot_is_rejected_and_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "match exactly"):
                with EpisodeStatistics(2, directory, False) as stats:
                    stats.add_batch(torch.tensor([0]), torch.ones(2), torch.ones(2), snapshot([1]))
            records, summary = read_results(directory)
            self.assertEqual(records, [])
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["stop_reason"], "ValueError")
            self.assertTrue(stats.log_file.closed)

    def test_existing_episode_logs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.log"
            path.write_text("existing\n")
            with self.assertRaises(FileExistsError), EpisodeStatistics(1, directory, False):
                self.fail("An existing log must not be overwritten")
            self.assertEqual(path.read_text(), "existing\n")


class PlayerIntegrationTests(unittest.TestCase):
    def test_exactly_100_episodes_across_four_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, 100)
            output = StringIO()
            with redirect_stdout(output):
                p.run()
            records, summary = read_results(directory)
            self.assertEqual(calls["steps"], 25)
            self.assertEqual(len(records), 100)
            self.assertEqual([r["episode"] for r in records], list(range(1, 101)))
            self.assertEqual(len((Path(directory) / "episodes.log").read_text().splitlines()), 100)
            self.assertEqual(sum(line.startswith("episode:") for line in output.getvalue().splitlines()), 100)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed_episodes"], 100)
            self.assertEqual(summary["mean_reward"], 2.5)
            self.assertEqual(summary["best_episode"]["episode"], 1)

    def test_partial_last_batch_does_not_pollute_best_or_semantic_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, 5)
            base_step = p.env_step
            def step(env, actions):
                obs, rewards, done, info = base_step(env, actions)
                if calls["steps"] == 2:
                    info["total_steps"] = torch.tensor([1., 999., 999., 999.])
                    info["semantic_contact_eval"]["semantic_success"][:3] = False
                return obs, rewards, done, info
            p.env_step = step
            with redirect_stdout(StringIO()):
                p.run()
            records, summary = read_results(directory)
            self.assertEqual(calls["steps"], 2)
            self.assertEqual([r["env_id"] for r in records], [0, 1, 2, 3, 0])
            self.assertEqual(summary["max_steps"], 1)
            self.assertEqual(summary["semantic_contact"]["episodes"], 5)
            self.assertEqual(summary["semantic_contact"]["semantic_sr_total"], 1)
            self.assertEqual(summary["semantic_contact"]["eval_frames"], 10)

    def test_custom_limit_single_env_and_no_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, 3, num_envs=1, semantics=False, episode_length=2)
            p.print_stats = False
            output = StringIO()
            with redirect_stdout(output):
                p.run()
            records, summary = read_results(directory)
            self.assertEqual(calls["steps"], 6)
            self.assertEqual([r["steps"] for r in records], [2., 2., 2.])
            self.assertEqual([r["reward"] for r in records], [2., 2., 2.])
            self.assertEqual(summary["best_episode"]["episode"], 1)
            self.assertEqual(output.getvalue(), "")

    def test_asynchronous_endings_preserve_reward_steps_and_rnn_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            p, _ = player(directory, 4, num_envs=3)
            p.is_rnn = True
            p.states = [torch.ones(1, 3, 1)]
            p.init_rnn = lambda: None
            states_at_action = []
            def action(obs, deterministic):
                states_at_action.append(p.states[0].clone())
                return torch.zeros(3, 1)
            p.get_action = action
            batches = iter([
                ([1., 10., 100.], [True, False, False], [101., 1., 1.], None),
                ([2., 20., 200.], [False, True, False], [1., 202., 2.], [0., 40., 0.]),
                ([3., 30., 300.], [True, False, True], [2., 1., 303.], None),
            ])
            def step(env, actions):
                rewards, done, lengths, totals = next(batches)
                ids = [i for i, ended in enumerate(done) if ended]
                info = {"total_steps": torch.tensor(lengths), "semantic_contact_eval": snapshot(ids[::-1])}
                if totals is not None:
                    info["total_rewards"] = torch.tensor(totals)
                return torch.zeros(3, 1), torch.tensor(rewards), torch.tensor(done), info
            p.env_step = step
            with redirect_stdout(StringIO()):
                p.run()
            records, summary = read_results(directory)
            self.assertEqual([(r["env_id"], r["reward"], r["steps"]) for r in records],
                             [(0, 1., 101.), (1, 40., 202.), (0, 5., 2.), (2, 600., 303.)])
            self.assertEqual(states_at_action[1].flatten().tolist(), [0., 1., 1.])
            self.assertEqual(states_at_action[2].flatten().tolist(), [0., 0., 1.])
            self.assertEqual(summary["best_episode"], records[3])

    def test_interrupt_keeps_completed_logs_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, 5, num_envs=1, save_rollouts=True)
            base_step = p.env_step
            def step(env, actions):
                if calls["steps"] == 1:
                    raise KeyboardInterrupt()
                return base_step(env, actions)
            p.env_step = step
            with redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
                p.run()
            records, summary = read_results(directory)
            self.assertEqual(len(records), 1)
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["stop_reason"], "KeyboardInterrupt")
            self.assertFalse(p.h5py_file.id.valid)

    def test_final_rollout_is_saved_without_extra_step(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, 4, num_envs=1, semantics=False, episode_length=2, save_rollouts=True)
            with redirect_stdout(StringIO()):
                p.run()
            self.assertEqual(calls["steps"], 8)
            self.assertFalse(p.h5py_file.id.valid)
            with h5py.File(str(Path(directory) / "rollouts.hdf5"), "r") as file:
                self.assertEqual(list(file["successful"]), ["rollout_0"])
                saved = file["successful/rollout_0"]
                self.assertEqual(saved["state"][-1, 0], 8)
                self.assertEqual(len(saved["state"]), len(saved["actions"]) + 1)

    def test_rollout_save_cap_does_not_end_finite_test(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, 6, num_envs=1, semantics=False, episode_length=2, save_rollouts=True)
            p.num_rollouts_to_save = 1
            with redirect_stdout(StringIO()):
                p.run()
            self.assertEqual(calls["steps"], 12)
            records, summary = read_results(directory)
            self.assertEqual(len(records), 6)
            self.assertEqual(summary["status"], "complete")
            with h5py.File(str(Path(directory) / "rollouts.hdf5"), "r") as file:
                self.assertEqual(len(file["successful"]), 1)

    def test_unlimited_mode_retains_legacy_logs_and_rollout_run_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            p, calls = player(directory, None, num_envs=1, semantics=False, episode_length=2, save_rollouts=True)
            output = StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit):
                p.run()
            self.assertEqual(calls["steps"], 2)
            self.assertEqual(output.getvalue().splitlines(), ["reward: 2.00 steps: 2.0"])
            self.assertFalse((Path(directory) / "episodes.log").exists())


class WiringTests(unittest.TestCase):
    def test_hydra_accepts_custom_limit_and_keeps_unlimited_default(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / "main/cfg"), version_base="1.1"):
            default = compose(config_name="config", overrides=["task=ResDexHand"])
            cfg = compose(config_name="config", overrides=["task=ResDexHand", "side=BiH", "test=true",
                                                           "test_episodes=7", "num_envs=4"])
        self.assertIsNone(default.test_episodes)
        self.assertEqual(cfg.test_episodes, 7)
        self.assertEqual(cfg.task.env.numEnvs, 4)
        self.assertFalse(cfg.task.env.training)

    def test_runner_passes_statistics_without_mutating_rollout_options(self):
        calls = []
        stub = SimpleNamespace(run=lambda: calls.append("run"))
        run = method("lib/rl/runner.py", "Runner", "run_play",
                     {"_restore": lambda *a, **kw: None, "_override_sigma": lambda *a: None})
        options = {"save_rollouts": False}
        def create(options):
            calls.append(options)
            return stub
        with redirect_stdout(StringIO()):
            run(SimpleNamespace(create_player=create),
                {"save_rollouts": options, "test_statistics": {"test_episodes": 7, "test_output_dir": "out"}})
        self.assertEqual(calls[0], dict(save_rollouts=False, test_episodes=7, test_output_dir="out"))
        self.assertEqual(options, {"save_rollouts": False})
        self.assertEqual(calls[1], "run")

    def test_video_cap_can_stop_recording_without_stopping_test(self):
        class Wrapper:
            def __init__(self, env):
                self.env = env
            def step(self, action):
                return self.env.step(action)
        tree = ast.parse((ROOT / "lib/utils/wandb_utils.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WandbVideoCaptureWrapper")
        logs = []
        namespace = {"gym": SimpleNamespace(Wrapper=Wrapper), "torch": torch,
                     "wandb": SimpleNamespace(Video=lambda *a, **kw: "video", log=logs.append)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "video", "exec"), namespace)
        env = SimpleNamespace(num_envs=1, camera_obs=torch.zeros(1, 1, 1, 4),
                              success_buf=torch.tensor([True]), failure_buf=torch.tensor([False]),
                              step=lambda action: ("obs", "reward", torch.tensor([True]), {}))
        cls = namespace["WandbVideoCaptureWrapper"]
        wrapper = cls(env, n_successful_videos_to_record=1, stop_after_videos=False)
        self.assertEqual(wrapper.step(None)[:2], ("obs", "reward"))
        wrapper.step(None)
        self.assertEqual(len(logs), 1)
        self.assertEqual(wrapper._videos, [[]])
        with self.assertRaises(SystemExit):
            cls(env, n_successful_videos_to_record=1).step(None)


if __name__ == "__main__":
    unittest.main()
