"""CPU regression tests; exercise real environment/player hooks without Isaac Gym."""
import ast
import json
import pickle
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import trimesh

from main.semantic_contact_eval import (
    CollisionSurfaceProxy, SemanticContactAccumulator, SemanticContactEvalConfig,
    SemanticContactSummary, format_semantic_contact_summary, full_fragment_end,
    load_semantic_regions, quat_rotate_xyzw,
    contact_required_mask_from_metadata, load_contact_required_masks,
    canonical_contact_metadata_path,
)
from main.semantic_contact_geometry import URDFCollisionKinematics


ROOT = Path(__file__).resolve().parents[1]
CFG = SemanticContactEvalConfig(enabled=True)


def extract_method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[method_name]


def accumulator(end_frames=(2,), starts=None, config=CFG, eligible=True, required=None):
    required = required if required is not None else [torch.ones(end + 1, dtype=torch.bool) for end in end_frames]
    result = SemanticContactAccumulator(torch.tensor(end_frames), config, required)
    result.reset(torch.arange(len(end_frames)), torch.tensor(starts or [0] * len(end_frames)), eligible)
    return result


class SemanticContactEvalTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertEqual(SemanticContactEvalConfig.from_mapping({}), SemanticContactEvalConfig())
        for config in ({"forceThreshold": -1}, {"contactDistance": float("nan")},
                       {"minRedContactRatio": 1.1}, {"envChunkSize": 0},
                       {"minContactFrames": 1.5}, {"pointsPerLink": True},
                       {"minPlannedSemanticContactCoverage": float("nan")},
                       {"expectedContactDistance": -0.1}):
            with self.assertRaises(ValueError):
                SemanticContactEvalConfig.from_mapping(config)

    def test_endpoint_uses_both_lengths_and_episode_cap(self):
        end = full_fragment_end(torch.tensor([100, 10, 2]), torch.tensor([80, 15, 3]), 50)
        self.assertEqual(end.tolist(), [46, 6, 0])

    def test_complete_success_and_terminal_inclusion(self):
        stats = accumulator()
        for t in range(3):
            stats.record(torch.tensor([t]), torch.tensor([32]), torch.tensor([0]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
        self.assertEqual(snap["eval_frames"].item(), 3)
        self.assertEqual(snap["red_events"].item(), 96)
        self.assertTrue(snap["complete"].item())
        self.assertTrue(snap["semantic_success"].item())

    def test_early_failure_counts_executed_frames(self):
        stats = accumulator((100,))
        stats.record(torch.tensor([0]), torch.tensor([32]), torch.tensor([0]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([False]))
        summary = SemanticContactSummary().add(snap, torch.tensor([0]))
        self.assertEqual(summary["red_contact_ratio"], 1)
        self.assertEqual(summary["eval_frames"], 1)
        self.assertEqual(summary["complete_episodes"], 0)
        self.assertEqual(summary["semantic_task_success_rate"], 0)

    def test_no_contact_and_zero_executed_frames_are_na(self):
        stats = accumulator((0, 0), starts=[0, 1])
        stats.record(torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
        snap = stats.finish(torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
        summary = SemanticContactSummary().add(snap, torch.tensor([[0], [1]]))
        self.assertIsNone(summary["red_contact_ratio"])
        self.assertIsNone(summary["black_contact_ratio"])
        self.assertEqual(summary["black_violation_frame_ratio"], 0)
        self.assertEqual(summary["no_contact_episodes"], 2)
        self.assertEqual(summary["semantic_task_success_rate"], 0)
        self.assertIn("red_contact_ratio: NA", format_semantic_contact_summary(summary))
        stats.reset(torch.tensor([1]), torch.tensor([1]))
        snap = stats.finish(torch.tensor([False, True]), torch.ones(2, dtype=torch.bool))
        summary = SemanticContactSummary().add(snap, torch.tensor([1]))
        self.assertIsNone(summary["black_violation_frame_ratio"])

    def test_random_nonzero_and_discontinuous_execution_fail(self):
        for start, eligible, frames in ((1, True, [1, 2]), (0, False, [0, 1, 2]),
                                        (0, True, [0, 2]), (0, True, [0, 1]),
                                        (0, True, [0, 1, 0, 2])):
            stats = accumulator(starts=[start], eligible=eligible)
            for t in frames:
                stats.record(torch.tensor([t]), torch.tensor([32]), torch.tensor([0]))
            snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
            self.assertFalse(snap["complete"].item())
            self.assertFalse(snap["semantic_success"].item())

    def test_duplicate_callbacks_and_finished_frames_not_counted(self):
        stats = accumulator((0,))
        for _ in range(2):
            stats.record(torch.tensor([0]), torch.tensor([32]), torch.tensor([0]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
        self.assertEqual(snap["red_events"].item(), 32)
        stats.record(torch.tensor([0]), torch.tensor([32]), torch.tensor([0]))
        self.assertIsNone(stats.finish(torch.tensor([True]), torch.tensor([True])))
        self.assertEqual(stats.eval_frames.item(), 1)

    def test_selective_reset_and_snapshot_survive(self):
        stats = accumulator((2, 2))
        stats.record(torch.tensor([0, 0]), torch.tensor([4, 6]), torch.tensor([0, 2]))
        snap = stats.finish(torch.tensor([True, False]), torch.tensor([False, False]))
        stats.reset(torch.tensor([0]), torch.tensor([0]))
        self.assertEqual(stats.red_events.tolist(), [0, 6])
        self.assertEqual(stats.black_events.tolist(), [0, 2])
        self.assertEqual(stats.eval_frames.tolist(), [0, 1])
        self.assertEqual(snap["red_events"].tolist(), [4])
        self.assertEqual(snap["env_ids"].tolist(), [0])

    def test_event_and_frame_weighted_aggregation_and_running_success(self):
        snap = dict(env_ids=torch.tensor([1, 3]), red_events=torch.tensor([90, 0]),
                    black_events=torch.tensor([10, 1]), eval_frames=torch.tensor([100, 2]),
                    black_frames=torch.tensor([1, 1]), complete=torch.tensor([True, False]),
                    planned_frames=torch.tensor([150, 10]), planned_required_frames=torch.tensor([50, 10]),
                    compliant_required_frames=torch.tensor([40, 1]), black_required_frames=torch.tensor([5, 0]),
                    missing_contact_frames=torch.tensor([5, 1]), unexecuted_required_frames=torch.tensor([0, 8]),
                    semantic_success=torch.tensor([True, False]))
        totals = SemanticContactSummary()
        summary = totals.add(snap, torch.tensor([3, 1]))
        self.assertAlmostEqual(summary["red_contact_ratio"], 90 / 101)
        self.assertAlmostEqual(summary["black_contact_ratio"], 11 / 101)
        self.assertAlmostEqual(summary["black_violation_frame_ratio"], 2 / 102)
        self.assertAlmostEqual(summary["operation_completion_ratio"], 102 / 160)
        self.assertAlmostEqual(summary["planned_semantic_contact_coverage"], 41 / 60)
        self.assertEqual(summary["semantic_task_success_rate"], 0.5)
        stats = accumulator()
        for t in range(3):
            stats.record(torch.tensor([t]), torch.tensor([32]), torch.tensor([0]))
        summary = totals.add(stats.finish(torch.tensor([True]), torch.tensor([True])), torch.tensor([0]))
        self.assertEqual(summary["semantic_task_success_rate"], 1)
        self.assertAlmostEqual(summary["semantic_sr_total"], 2 / 3)
        with self.assertRaises(ValueError):
            totals.add(snap, torch.tensor([1, 2]))

    def test_each_success_constraint_is_required(self):
        for nr, nb, black_frames, contact_frames, task_success in (
            (90, 10, 0, 3, True), (89, 0, 0, 2, True),
            (100, 0, 1, 3, True), (100, 0, 0, 3, False), (0, 0, 0, 0, True),
        ):
            stats = accumulator()
            for t in range(3):
                stats.record(torch.tensor([t]), torch.tensor([0]), torch.tensor([0]))
            stats.red_events[0], stats.black_events[0] = nr, nb
            stats.black_frames[0], stats.contact_frames[0] = black_frames, contact_frames
            snap = stats.finish(torch.tensor([True]), torch.tensor([task_success]))
            self.assertFalse(snap["semantic_success"].item())
        # Isolate red threshold and verify inclusive threshold comparison.
        for nr, expected in ((89, False), (90, True)):
            config = replace(CFG, maxBlackContactRatio=1, maxBlackViolationFrameRatio=1,
                             minPlannedSemanticContactCoverage=0)
            stats = accumulator(config=config)
            for t in range(3):
                stats.record(torch.tensor([t]), torch.tensor([nr]), torch.tensor([100 - nr]))
            snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
            self.assertEqual(snap["semantic_success"].item(), expected)

    def make_detector(self, points, regions=None, scales=None, config=CFG, body_ids=None):
        return CollisionSurfaceProxy(
            torch.tensor(points, dtype=torch.float32), torch.tensor(body_ids or [0] * len(points)),
            regions or [(torch.tensor([[0., 0, 0]]), torch.tensor([[0.02, 0, 0]]))],
            torch.zeros(len(scales or [1]), dtype=torch.long), torch.tensor(scales or [1.]), config,
        )

    def assert_required_partition(self, snapshot):
        actual = sum(snapshot[name] for name in ("compliant_required_frames", "black_required_frames",
                                                 "missing_contact_frames", "unexecuted_required_frames"))
        self.assertTrue(torch.equal(snapshot["planned_required_frames"], actual))

    def test_short_red_only_failure_keeps_complete_plan_denominators(self):
        stats = accumulator((299,))
        for frame in range(36):
            stats.record(torch.tensor([frame]), torch.tensor([32]), torch.tensor([0]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([False]))
        result = SemanticContactSummary().add(snap, torch.tensor([0]))
        self.assertEqual(result["red_contact_ratio"], 1)
        self.assertEqual(result["black_contact_ratio"], 0)
        self.assertEqual(result["operation_completion_ratio"], 0.12)
        self.assertEqual(result["planned_semantic_contact_coverage"], 0.12)
        self.assertEqual(result["unexecuted_required_frames"], 264)
        self.assertEqual(result["black_required_frames"], 0)
        self.assertEqual(result["semantic_task_success_rate"], 0)
        self.assert_required_partition(snap)

    def test_required_partition_separates_approach_release_missing_black_unexecuted(self):
        stats = accumulator((7,), required=[[False, True, True, True, False, True, True, False]])
        # No contact while approaching/releasing is not missing contact.
        for frame, red, black in [(0, 0, 0), (1, 32, 0), (2, 32, 1), (3, 0, 0), (4, 0, 0)]:
            stats.record(torch.tensor([frame]), torch.tensor([red]), torch.tensor([black]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([False]))
        for name, expected in [("planned_required_frames", 5), ("compliant_required_frames", 1),
                               ("black_required_frames", 1), ("missing_contact_frames", 1),
                               ("unexecuted_required_frames", 2), ("black_frames", 1)]:
            self.assertEqual(snap[name].item(), expected)
        self.assert_required_partition(snap)

    def test_nonzero_start_skips_duplicates_and_revisited_frames_preserve_partition(self):
        stats = accumulator((5,), starts=[2])
        for frame in (2, 2, 4, 4, 3, 5, 3):
            stats.record(torch.tensor([frame]), torch.tensor([1]), torch.tensor([0]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
        self.assertEqual(snap["eval_frames"].item(), 4)
        self.assertEqual(snap["unexecuted_required_frames"].item(), 2)
        self.assertFalse(snap["complete"].item())
        self.assertFalse(snap["semantic_success"].item())
        self.assert_required_partition(snap)

    def test_empty_required_mask_coverage_na_even_with_red_contacts(self):
        stats = accumulator(required=[[False] * 3])
        for frame in range(3):
            stats.record(torch.tensor([frame]), torch.tensor([32]), torch.tensor([0]))
        snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
        result = SemanticContactSummary().add(snap, torch.tensor([0]))
        self.assertEqual(result["operation_completion_ratio"], 1)
        self.assertIsNone(result["planned_semantic_contact_coverage"])
        self.assertEqual(result["semantic_task_success_rate"], 0)
        self.assertIn("planned_semantic_contact_coverage: NA", format_semantic_contact_summary(result))
        self.assert_required_partition(snap)

    def test_required_no_contact_coverage_zero_and_minimum_inclusive(self):
        for compliant_count, expected in ((0, False), (3, False), (4, True)):
            stats = accumulator((4,))
            for frame in range(5):
                stats.record(torch.tensor([frame]), torch.tensor([32 if frame < compliant_count else 0]), torch.tensor([0]))
            snap = stats.finish(torch.tensor([True]), torch.tensor([True]))
            result = SemanticContactSummary().add(snap, torch.tensor([0]))
            self.assertEqual(result["planned_semantic_contact_coverage"], compliant_count / 5)
            self.assertEqual(bool(snap["semantic_success"].item()), expected)
            self.assertEqual(result["missing_contact_frames"], 5 - compliant_count)
            self.assert_required_partition(snap)

    def test_required_state_selective_reset_keeps_plan_and_snapshot(self):
        stats = accumulator((2, 2), required=[[False, True, True], [True, True, False]])
        for frame in (0, 1):
            stats.record(torch.tensor([frame, frame]), torch.tensor([32, 0]), torch.tensor([0, 1]))
        snap = stats.finish(torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
        plan = stats.contact_required.clone()
        stats.reset(torch.tensor([0]), torch.tensor([0]))
        self.assertTrue(torch.equal(plan, stats.contact_required))
        self.assertEqual(stats.executed_required_frames.tolist(), [0, 2])
        self.assertEqual(stats.black_required_frames.tolist(), [0, 2])
        self.assertEqual(stats.compliant_required_frames.tolist(), [0, 0])
        self.assertEqual(snap["compliant_required_frames"].item(), 1)
        self.assertEqual(snap["unexecuted_required_frames"].item(), 1)
        self.assert_required_partition(snap)

    def test_source_intent_local_offset_precedence_and_old_grasp_interval(self):
        metadata = {"contact_retarget": {"grasp_interval": np.array([2, 5]),
                    "source_contact_intent": {"contact_frame_mask": np.array([True, False, True, False])}}}
        mask = contact_required_mask_from_metadata(metadata, 7)
        self.assertEqual(mask.tolist(), [False, False, True, False, True, False, False])
        self.assertEqual(contact_required_mask_from_metadata(metadata, 4).tolist(), [False, False, True, False])
        del metadata["contact_retarget"]["source_contact_intent"]
        self.assertEqual(contact_required_mask_from_metadata(metadata, 7).tolist(),
                         [False, False, True, True, True, True, False])
        self.assertIsNone(contact_required_mask_from_metadata({}, 7))
        self.assertEqual(contact_required_mask_from_metadata({"grasp_interval": [8, 10]}, 7).sum().item(), 0)

    def test_invalid_metadata_does_not_silently_infer_required_frames(self):
        for metadata in ({"source_contact_intent": {"contact_frame_mask": [True]}},
                         {"grasp_interval": [-1, 2]}, {"grasp_interval": [3, 2]},
                         {"grasp_interval": [0.5, 2]}, {"grasp_interval": [False, True]},
                         {"grasp_interval": [2, 3], "source_contact_intent": {"contact_frame_mask": [True]}},
                         {"grasp_interval": [2, 3], "source_contact_intent": {"contact_frame_mask": [0, 1]}}):
            with self.assertRaises(ValueError):
                contact_required_mask_from_metadata(metadata, 10)

    def test_shared_json_priority_and_same_mask_for_different_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "required.json"
            path.write_text(json.dumps({"seq": {"required_intervals": [[1, 2], [4, 4]]},
                                        "other": [False, True, False, True, False, True]}))
            config = replace(CFG, contactRequiredMaskFile=str(path))
            output = StringIO()
            with redirect_stdout(output):
                a, sources_a = load_contact_required_masks(["seq", "other"], [5, 5], torch.zeros(2, 5, 5), config,
                                                           {"seq": "not_read.pkl"})
                b, sources_b = load_contact_required_masks(["seq", "other"], [5, 5], torch.ones(2, 5, 5), config)
            self.assertEqual(a[0].tolist(), [False, True, True, False, True])
            self.assertEqual(a[1].tolist(), [False, True, False, True, False])
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
            self.assertEqual(sources_a, sources_b)
            self.assertNotIn("FALLBACK", output.getvalue())

    def test_shared_json_rejects_missing_short_numeric_or_invalid_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "required.json"
            config = replace(CFG, contactRequiredMaskFile=str(path))
            for entry in ({}, {"seq": [False, True]}, {"seq": [0, 1, 0]},
                          {"seq": {"required_intervals": [[2, 1]]}},
                          {"seq": {"required_intervals": [[0, 1.5]]}}):
                path.write_text(json.dumps(entry))
                with self.assertRaises(ValueError):
                    load_contact_required_masks(["seq"], [3], torch.zeros(1, 3, 5), config)

    def test_metadata_loader_and_explicit_reference_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "common.pkl"
            original = {"contact_retarget": {"grasp_interval": [1, 3],
                       "source_contact_intent": {"contact_frame_mask": [True, False, True]}},
                       "opt_wrist_pos": np.ones((5, 3))}
            path.write_bytes(pickle.dumps(original))
            before = path.read_bytes()
            output = StringIO()
            with redirect_stdout(output):
                masks, sources = load_contact_required_masks(["seq"], [5], None, CFG, {"seq": path})
            self.assertEqual(masks[0].tolist(), [False, True, False, True, False])
            self.assertIn("source_contact_intent.contact_frame_mask", sources[0])
            self.assertEqual(path.read_bytes(), before)
            # Older common PKLs retain a coarse interval, rather than per-policy inference.
            del original["contact_retarget"]["source_contact_intent"]
            path.write_bytes(pickle.dumps(original))
            with redirect_stdout(output):
                masks, sources = load_contact_required_masks(["seq"], [5], None, CFG, {"seq": path})
            self.assertEqual(masks[0].tolist(), [False, True, True, True, False])
            self.assertIn("grasp_interval (coarse)", sources[0])
            distances = torch.full((1, 5, 5), 1.)
            distances[0, 1, 2] = CFG.expectedContactDistance
            with redirect_stdout(output):
                masks, sources = load_contact_required_masks(["seq"], [5], distances, CFG)
            self.assertEqual(masks[0].tolist(), [False, True, False, False, False])
            self.assertIn("FALLBACK checkpoint reference tips_distance", output.getvalue())

    def test_canonical_metadata_lookup_normalizes_tool_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "TACO/mano2artimano_rh/seq/004/seq.pkl"
            path.parent.mkdir(parents=True)
            path.touch()
            self.assertEqual(canonical_contact_metadata_path(root, "TACO", "artimano_rh", "seq", 4), path)
            self.assertIsNone(canonical_contact_metadata_path(root, "TACO", "artimano_rh", "seq", None))
            self.assertIsNone(canonical_contact_metadata_path(root, "TACO", "artimano_rh", "missing", 4))
            (path.parent / "extra.pkl").touch()
            with self.assertRaises(ValueError):
                canonical_contact_metadata_path(root, "TACO", "artimano_rh", "seq", "004")
            with self.assertRaises(ValueError):
                canonical_contact_metadata_path(root, "TACO", "artimano_rh", "../seq", "004")

    def test_force_distance_and_conservative_boundary(self):
        points = [[0, 0, 0], [0.02, 0, 0], [0.01, 0, 0], [0.1, 0, 0]]
        detector = self.make_detector(points, scales=[1., 1., 1.], config=replace(CFG, envChunkSize=1))
        bodies = torch.zeros(3, 1, 13)
        bodies[..., 6] = 1
        tool = bodies[:, 0].clone()
        forces = torch.tensor([[[1., 0, 0]], [[0.999, 0, 0]], [[float("nan"), 0, 0]]])
        nr, nb = detector.detect(bodies, forces, tool, torch.ones(3, dtype=torch.bool))
        self.assertEqual(nr.tolist(), [1, 0, 0])
        self.assertEqual(nb.tolist(), [2, 0, 0])
        detector = self.make_detector([[0, 0, 0]], regions=[(
            torch.tensor([[0.005, 0, 0]]), torch.tensor([[0.0050005, 0, 0]]))])
        nr, nb = detector.detect(bodies[:1], forces[:1], tool[:1], torch.tensor([True]))
        self.assertEqual((nr.item(), nb.item()), (0, 1))

    def test_rotation_translation_and_real_tool_scale(self):
        detector = self.make_detector([[0, 0.04, 0]], scales=[2.])
        bodies = torch.zeros(1, 1, 13)
        bodies[..., 6] = 1
        bodies[..., :3] = torch.tensor([1., 2, 3])
        tool = bodies[:, 0].clone()
        tool[:, 3:7] = torch.tensor([0., 0, 2 ** -0.5, 2 ** -0.5])
        before = bodies.clone(), tool.clone()
        nr, nb = detector.detect(bodies, torch.tensor([[[1., 0, 0]]]), tool, torch.tensor([True]))
        self.assertEqual((nr.item(), nb.item()), (0, 1))
        self.assertTrue(torch.equal(bodies, before[0]))
        self.assertTrue(torch.equal(tool, before[1]))
        q = tool[:, 3:7]
        actual = quat_rotate_xyzw(q, torch.tensor([[1., 0, 0]]))
        torch.testing.assert_close(actual, torch.tensor([[0., 1, 0]]), atol=1e-6, rtol=0)

    def test_per_link_force_all_links_and_chunk_equivalence(self):
        detector = self.make_detector([[0, 0, 0], [0, 0, 0]], body_ids=[0, 1], scales=[1., 1., 1.])
        bodies = torch.zeros(3, 2, 13)
        bodies[..., 6] = 1
        bodies[:, 1, 0] = 0.02
        forces = torch.zeros(3, 2, 3)
        forces[0, 0, 0], forces[1, 1, 0], forces[2, :, 0] = 1, 1, 1
        tool = bodies[:, 0].clone()
        nr, nb = detector.detect(bodies, forces, tool, torch.tensor([True, True, False]))
        self.assertEqual(nr.tolist(), [1, 0, 0])
        self.assertEqual(nb.tolist(), [0, 1, 0])
        detector.config = replace(CFG, envChunkSize=1)
        nr2, nb2 = detector.detect(bodies, forces, tool, torch.tensor([True, True, False]))
        self.assertTrue(torch.equal(nr, nr2) and torch.equal(nb, nb2))

    def test_multiple_tools_keep_separate_regions(self):
        detector = CollisionSurfaceProxy(torch.zeros(1, 3), torch.tensor([0]),
            [(torch.zeros(1, 3), torch.tensor([[1., 0, 0]])),
             (torch.tensor([[1., 0, 0]]), torch.zeros(1, 3))],
            torch.tensor([0, 1]), torch.ones(2), CFG)
        bodies = torch.zeros(2, 1, 13)
        bodies[..., 6] = 1
        nr, nb = detector.detect(bodies, torch.ones(2, 1, 3), bodies[:, 0], torch.ones(2, dtype=torch.bool))
        self.assertEqual(nr.tolist(), [1, 0])
        self.assertEqual(nb.tolist(), [0, 1])

    def test_semantic_mesh_sampling_and_missing_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool.ply"
            mesh = trimesh.creation.box()
            colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
            colors[:, 3] = 255
            colors[:4, 0] = 255
            mesh.visual.vertex_colors = colors
            mesh.export(path)
            red, black = load_semantic_regions(path)
            self.assertEqual(red.shape, (512, 3))
            self.assertEqual(black.shape, (512, 3))
            self.assertTrue(torch.equal(red, load_semantic_regions(path)[0]))
            mesh.visual.vertex_colors = np.full_like(colors, 200)
            mesh.export(path)
            with self.assertRaises(ValueError):
                load_semantic_regions(path)

    def test_urdf_collision_origin_mesh_scale_and_rng_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trimesh.creation.box(extents=[1, 1, 1]).export(directory / "box.stl")
            urdf = directory / "hand.urdf"
            urdf.write_text('''<robot name="hand">
              <link name="palm"><collision><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
                <geometry><mesh filename="box.stl" scale="2 1 1"/></geometry></collision></link>
              <link name="finger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
              <joint name="j" type="fixed"><parent link="palm"/><child link="finger"/></joint>
            </robot>''')
            state = np.random.get_state()
            model = URDFCollisionKinematics(urdf, [], 32)
            self.assertTrue(np.array_equal(np.random.get_state()[1], state[1]))
            self.assertEqual(model.point_link_names, ["palm", "finger"])
            self.assertEqual(model.local_points.shape, (64, 3))
            # Origin rotation maps scaled x half-extent 1 onto y, y half-extent .5 onto x.
            points = model.link_points["palm"]
            self.assertTrue((np.abs(points[:, 0] - 1) <= 0.5 + 1e-6).all())
            self.assertTrue((np.abs(points[:, 1]) <= 1 + 1e-6).all())
            self.assertTrue(np.allclose(points, URDFCollisionKinematics(urdf, [], 32).link_points["palm"]))
            with self.assertRaises(ValueError):
                CollisionSurfaceProxy.from_urdf(urdf, [], {"palm": 0}, [], torch.tensor([0]), torch.ones(1), CFG)

    def test_test_only_init_is_noop_when_disabled_or_training(self):
        init = extract_method("maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv", "_init_semantic_contact_eval")
        for training, enabled in ((True, True), (True, False), (False, False)):
            env = SimpleNamespace(training=training, cfg={"env": {"semanticContactEval": {"enabled": enabled}}})
            init(env)  # No simulator/dataset access may be attempted.

    def test_environment_hook_collects_before_increment_preserves_policy(self):
        post = extract_method("maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv", "post_physics_step")
        events = []
        stats = accumulator((0,))
        action = torch.tensor([[0.4]])
        env = SimpleNamespace(progress_buf=torch.tensor([0]), running_progress_buf=torch.tensor([0]),
            randomize_buf=torch.tensor([0]), reset_buf=torch.tensor([True]), success_buf=torch.tensor([True]),
            actions=action, semantic_contact_eval=stats, _rigid_body_state=None, net_cf=None,
            _manip_obj_rh_root_state=None,
            compute_observations=lambda: events.append("obs"),
            compute_reward=lambda actions: events.append(("reward", actions.clone())))
        def detect(*args):
            self.assertEqual(env.progress_buf.item(), 0)
            events.append("detect")
            return torch.tensor([32]), torch.tensor([0])
        env.semantic_contact_detector = SimpleNamespace(detect=detect)
        post(env)
        self.assertEqual([event if isinstance(event, str) else event[0] for event in events], ["obs", "reward", "detect"])
        self.assertTrue(torch.equal(env.actions, action))
        self.assertEqual(env._semantic_contact_snapshot["eval_frames"].item(), 1)
        self.assertEqual(env.progress_buf.item(), 1)
        env.semantic_contact_eval = None
        events.clear()
        post(env)
        self.assertEqual(len(events), 2)
        self.assertEqual(env.progress_buf.item(), 2)

    def test_environment_reset_hook_clears_only_selected_env(self):
        reset = extract_method("maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv", "reset_idx")
        stats = accumulator((2, 2))
        stats.record(torch.tensor([0, 0]), torch.tensor([4, 6]), torch.tensor([0, 2]))
        env = SimpleNamespace(_refresh=lambda: None, randomize=False, training=False,
            gym=SimpleNamespace(get_frame_count=lambda sim: 0), sim=None,
            _reset_default=lambda ids: None, semantic_contact_eval=stats,
            progress_buf=torch.zeros(2, dtype=torch.long), random_state_init=False)
        reset(env, torch.tensor([0]))
        self.assertEqual(stats.red_events.tolist(), [0, 6])
        self.assertEqual(stats.eval_frames.tolist(), [0, 1])

    def test_enabled_initializer_uses_all_collision_links_and_actual_actor_scale(self):
        init = extract_method("maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv", "_init_semantic_contact_eval")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            urdf = directory / "hand.urdf"
            urdf.write_text('''<robot name="hand">
              <link name="palm"><collision><geometry><box size=".01 .01 .01"/></geometry></collision></link>
              <link name="finger"><collision><geometry><sphere radius=".01"/></geometry></collision></link>
              <joint name="j" type="fixed"><parent link="palm"/><child link="finger"/></joint>
            </robot>''')
            path = directory / "semantic.ply"
            mesh = trimesh.creation.box()
            colors = np.zeros((8, 4), dtype=np.uint8)
            colors[:, 3] = 255
            colors[:4, 0] = 255
            mesh.visual.vertex_colors = colors
            mesh.export(path)
            requested_links = []
            def handle(env, actor, name):
                requested_links.append(name)
                return {"palm": 0, "finger": 1}[name]
            env = SimpleNamespace(training=False,
                cfg={"env": {"semanticContactEval": {"enabled": True,
                    "semanticMeshPaths": {"seq": str(path)}}}},
                gym=SimpleNamespace(find_actor_handle=lambda env, name: 0,
                    find_actor_rigid_body_handle=handle, get_actor_scale=lambda env, actor: env + 1.),
                dexhand_rh=SimpleNamespace(urdf_path=urdf, dof_names=[], body_names=["finger"]),
                envs=[0, 1], num_envs=2, new_obj=None, dataIndices=["seq"], device="cpu", max_episode_length=100,
                demo_data_rh={"obj_mesh_path": ["not_loaded.ply"] * 2, "seq_len": torch.tensor([10, 12]),
                              "tips_distance": torch.zeros(2, 12, 5)},
                demo_data_lh={"seq_len": torch.tensor([12, 10])})
            init(env)
            self.assertEqual(requested_links, ["palm", "finger"])
            self.assertEqual(env.semantic_contact_detector.scales.tolist(), [1., 2.])
            self.assertEqual(len(env.semantic_contact_detector.regions), 1)
            self.assertEqual(env.semantic_contact_detector.body_indices.bincount().tolist(), [32, 32])
            self.assertEqual(env.semantic_contact_eval.end_frames.tolist(), [6, 6])
            # Exercise the real test initializer with shared correction metadata.
            class Hand:
                urdf_path = urdf
                dof_names = []
                body_names = ["finger"]
                def __str__(self):
                    return "artimano_rh"
            env.dexhand_rh = Hand()
            env.new_obj = 4
            init.__globals__["ManipDataFactory"] = SimpleNamespace(dataset_type=lambda idx: "taco")
            common = directory / "common/TACO/mano2artimano_rh/seq/004/seq.pkl"
            common.parent.mkdir(parents=True)
            common.write_bytes(pickle.dumps({"contact_retarget": {"grasp_interval": [2, 4],
                "source_contact_intent": {"contact_frame_mask": [True, False, True]}},
                "opt_wrist_pos": np.full((10, 3), 999.)}))
            eval_config = env.cfg["env"]["semanticContactEval"]
            eval_config["contactRetargetRoot"] = str(directory / "common")
            reference = env.demo_data_rh["tips_distance"].clone()
            output = StringIO()
            with redirect_stdout(output):
                init(env)
            self.assertEqual(env.semantic_contact_eval.contact_required.tolist(),
                             [[False, False, True, False, True, False, False]] * 2)
            self.assertTrue(torch.equal(env.demo_data_rh["tips_distance"], reference))
            self.assertNotIn("opt_wrist_pos", env.demo_data_rh)
            self.assertIn("source_contact_intent.contact_frame_mask", output.getvalue())
            # JSON also bypasses a broken explicitly configured PKL.
            shared = directory / "shared.json"
            shared.write_text(json.dumps({"seq": {"required_intervals": [[1, 1]]}}))
            eval_config["contactRequiredMaskFile"] = str(shared)
            eval_config["contactMetadataPaths"] = {"seq": "not_read.pkl"}
            with redirect_stdout(output):
                init(env)
            self.assertEqual(env.semantic_contact_eval.planned_required_frames.tolist(), [1, 1])
            self.assertTrue(torch.equal(env.demo_data_rh["tips_distance"], reference))

    def test_player_logs_original_prefix_batch_and_cumulative(self):
        run = extract_method("lib/rl/player.py", "MyBasePlayer", "run")
        stats = accumulator((2, 2))
        for t in range(3):
            stats.record(torch.tensor([t, t]), torch.tensor([32, 0]), torch.tensor([0, 0]))
        first = stats.finish(torch.ones(2, dtype=torch.bool), torch.tensor([True, False]))
        stats.reset(torch.tensor([1]), torch.tensor([0]))
        for t in range(3):
            stats.record(torch.tensor([t, t]), torch.tensor([0, 32]), torch.tensor([0, 0]))
        second = stats.finish(torch.tensor([False, True]), torch.tensor([False, True]))
        for enabled in (False, True):
            batches = iter([(torch.ones(2, dtype=torch.bool), first), (torch.tensor([False, True]), second)])
            def step(env, actions):
                done, snap = next(batches)
                info = {"total_steps": torch.tensor([101., 101.])}
                if enabled:
                    info["semantic_contact_eval"] = snap
                return torch.zeros(2, 1), torch.ones(2), done, info
            player = SimpleNamespace(render_env=False, is_deterministic=True, save_rollouts=False,
                is_rnn=False,
                env=SimpleNamespace(success_buf=torch.ones(2, dtype=torch.bool), reset_done=lambda: (torch.zeros(2, 1), None)),
                wait_for_checkpoint=lambda: None, env_reset=lambda env: torch.zeros(2, 1),
                get_batch_size=lambda obs, batch: 2, get_action=lambda obs, det: torch.zeros(2, 1),
                env_step=step, print_stats=True)
            output = StringIO()
            with redirect_stdout(output), self.assertRaises(StopIteration):
                run(player)
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 2)
            if enabled:
                self.assertTrue(lines[0].startswith("reward: 1.00 steps: 101.0 operation_completion_ratio: 1.0000 "
                                                   "planned_semantic_contact_coverage: 0.5000 black_violation_frame_ratio: 0.0000 "))
                self.assertIn("semantic_task_success_rate: 0.5000 semantic_sr_total: 0.5000", lines[0])
                self.assertIn("eval_frames: 6 complete_episodes: 2/2 no_contact_episodes: 1", lines[0])
                self.assertIn("semantic_task_success_rate: 1.0000 semantic_sr_total: 0.6667", lines[1])
                self.assertIn("planned_frames: 6 required_frames: 6 compliant_required_frames: 3 "
                              "missing_contact_frames: 3 unexecuted_required_frames: 0", lines[0])
                self.assertGreater(lines[0].index("red_contact_ratio:"), lines[0].index("semantic_task_success_rate:"))
            else:
                self.assertEqual(lines, ["reward: 1.00 steps: 101.0"] * 2)


if __name__ == "__main__":
    unittest.main()
