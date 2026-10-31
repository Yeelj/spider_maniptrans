"""CPU regressions for SPIDER's structural contract and legacy packing."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lib.utils.spider_reference import (
    find_manifest, name_order, normalize_tool, object_transform,
    read_bundle, resolve_asset, sha256, validate_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    m = {"schema_version": 1, "source_dataset": "TACO", "task": "test_task", "task_aliases": ["exact_alias"],
         "new_obj": "045", "robot_type": "artimano", "recorded_frames": 4, "expected_frames": 4,
         "complete": True, "fps": 60.0, "frame_coordinates": "maniptrans_world",
         "rotation_convention": "rotvec", "state_quaternion": "xyzw", "validation_status": "failed",
         "validation_passed": False, "trajectory_file": "trajectory.npz"}
    arrays = {"frame_index": np.arange(4), "time_seconds": np.arange(4) / 60}
    for side in ("rh", "lh"):
        m[side] = {"dof_names": ["q" + str(i) for i in range(22)],
                   "body_names": ["palm"] + ["link" + str(i) for i in range(27)],
                   "source_object_id": "205" if side == "rh" else "143",
                   "selected_asset_id": "045" if side == "rh" else "143",
                   "object_scale": 1.0, "object_mass_override": None,
                   "object_density": 200.0, "object_friction": 2.0,
                   "object_rolling_friction": 0.05, "object_torsion_friction": 0.05,
                   "obj_mesh_path": "mesh.ply", "obj_urdf_path": "obj.urdf", "hand_urdf": "hand.urdf",
                   "hand_urdf_sha256": "test"}
        arrays[side + "_wrist_pos"] = np.zeros((4, 3), dtype=np.float32)
        arrays[side + "_wrist_rotvec"] = np.zeros((4, 3), dtype=np.float32)
        arrays[side + "_dof_pos"] = np.zeros((4, 22), dtype=np.float32)
        arrays[side + "_joint_pos"] = np.zeros((4, 28, 3), dtype=np.float32)
        arrays[side + "_source_object_transform"] = np.repeat(np.eye(4)[None], 4, axis=0)
        arrays[side + "_sim_object_state"] = np.zeros((4, 13), dtype=np.float32)
        arrays[side + "_sim_object_state"][:, 6] = 1
        arrays[side + "_sim_object_state"][:, 2] = -10  # failed dropped object
    return m, arrays


def pack_method():
    path = ROOT / "maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DexHandManipBiHEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "pack_data")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["pack_data"]


class BundleTests(unittest.TestCase):
    def test_failed_but_complete_is_allowed_and_strict_policy_rejects(self):
        m, a = fixture()
        validate_bundle(m, a)
        with self.assertRaises(ValueError):
            validate_bundle(m, a, allow_failed=False)
        self.assertFalse(m["validation_passed"])

    def test_incomplete_and_nonfinite_never_allowed(self):
        for key, value in (("complete", False), ("expected_frames", 5), ("fps", float("nan"))):
            m, a = fixture()
            m[key] = value
            with self.assertRaises(ValueError):
                validate_bundle(m, a, allow_failed=True)
        m, a = fixture()
        for key in ("lh_dof_pos", "rh_joint_pos", "rh_wrist_pos", "lh_sim_object_state"):
            changed = deepcopy(a)
            changed[key].flat[0] = np.nan
            with self.assertRaises(ValueError):
                validate_bundle(m, changed)

    def test_time_axis_shapes_rotations_and_palm_are_validated(self):
        m, a = fixture()
        for key, value in (("frame_index", np.array([0, 1, 1, 3])),
                           ("time_seconds", np.arange(4) / 30),
                           ("lh_dof_pos", np.zeros((3, 22))),
                           ("rh_joint_pos", np.ones((4, 28, 3))),
                           ("rh_source_object_transform", np.ones((4, 4, 4))),
                           ("lh_sim_object_state", np.zeros((4, 13)))):
            changed = dict(a, **{key: value})
            with self.assertRaises(ValueError):
                validate_bundle(m, changed)

    def test_link_name_alignment_handles_reorder_and_rejects_duplicate_or_unknown(self):
        self.assertEqual(name_order(["b", "palm", "a"], ["palm", "a", "b"], "test"), [1, 2, 0])
        for source, target in ((["a", "a"], ["a", "b"]), (["a", "c"], ["a", "b"]), (["a", "b"], ["a", "a"])):
            with self.assertRaises(ValueError):
                name_order(source, target, "test")

    def test_object_policy_does_not_follow_failed_rollout_by_default(self):
        m, a = fixture()
        np.testing.assert_array_equal(object_transform(a, "rh", "demonstration")[:, 2, 3], 0)
        np.testing.assert_array_equal(object_transform(a, "rh", "spider_rollout")[:, 2, 3], -10)
        with self.assertRaises(ValueError):
            object_transform(a, "rh", "unknown")

    def test_namespace_normalization(self):
        self.assertEqual(normalize_tool(45, "TACO"), "045")
        self.assertEqual(normalize_tool("kni@1", "OakInk-v2"), "kni@1")
        self.assertEqual(normalize_tool(None, "TACO"), "original")
        for value in ("../045", "45/other", -1, 1000):
            with self.assertRaises(ValueError):
                normalize_tool(value, "TACO")

    def test_no_fallback_exact_alias_and_checksum(self):
        m, a = fixture()
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "045/TACO/training/test_task"
            task.mkdir(parents=True)
            np.savez(str(task / "trajectory.npz"), **a)
            m["trajectory_sha256"] = sha256(task / "trajectory.npz")
            (task / "manifest.json").write_text(json.dumps(m))
            path = find_manifest(directory, 45, "TACO", "exact_alias")
            loaded, arrays = read_bundle(path)
            self.assertFalse(arrays["rh_dof_pos"].flags.writeable)
            self.assertFalse(loaded["validation_passed"])
            with self.assertRaises(FileNotFoundError):
                find_manifest(directory, 153, "TACO", "test_task")
            with self.assertRaises(FileNotFoundError):
                find_manifest(directory, 45, "TACO", "test")
            np.savez(str(task / "trajectory.npz"), corrupted=np.ones(2))
            with self.assertRaises(ValueError):
                read_bundle(path)

    def test_ambiguous_alias_is_rejected(self):
        m, a = fixture()
        with tempfile.TemporaryDirectory() as directory:
            for name in ("one", "two"):
                path = Path(directory) / "045/TACO/training" / name
                path.mkdir(parents=True)
                (path / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(FileNotFoundError):
                find_manifest(directory, "045", "TACO", "exact_alias")

    def test_asset_relocation_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tool.ply").touch()
            self.assertEqual(resolve_asset("/old/data/tool.ply", root / "manifest.json", path_map={"/old/data": directory}), str(root / "tool.ply"))
            with self.assertRaises(FileNotFoundError):
                resolve_asset("/wrong/tool.ply", root / "manifest.json")


class PackingTests(unittest.TestCase):
    def test_robot_links_ignore_poisoned_mano_and_preserve_single_batch_axis(self):
        hand = SimpleNamespace(body_names=["palm", "index1y", "index1z"])
        env = SimpleNamespace(device="cpu", max_episode_length=10, dexhand_rh=hand)
        links = torch.arange(4 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3)
        data = {"reference_source": "spider", "obj_trajectory": torch.eye(4).repeat(4, 1, 1),
                "reference_joint_pos": links, "reference_joint_velocity": links + 10,
                "mano_joints": {"bad": torch.ones(4, 3) * 999}, "obj_verts": torch.zeros(1000, 3)}
        packed = pack_method()(env, [data])
        self.assertEqual(packed["mano_joints"].shape, (1, 4, 6))
        self.assertEqual(packed["obj_verts"].shape, (1, 1000, 3))
        torch.testing.assert_close(packed["mano_joints"][0], links[:, 1:].reshape(4, -1))
        self.assertFalse(torch.equal(packed["mano_joints"][0, :, :3], packed["mano_joints"][0, :, 3:]))

    def test_padding_keeps_real_lengths_and_does_not_mutate_reference(self):
        hand = SimpleNamespace(body_names=["palm", "link"])
        env = SimpleNamespace(device="cpu", max_episode_length=10, dexhand_rh=hand)
        samples = [{"reference_source": "spider", "obj_trajectory": torch.eye(4).repeat(n, 1, 1),
                    "reference_joint_pos": torch.zeros(n, 2, 3),
                    "reference_joint_velocity": torch.zeros(n, 2, 3)} for n in (2, 5)]
        packed = pack_method()(env, samples)
        self.assertEqual(packed["seq_len"].tolist(), [2, 5])
        self.assertEqual(samples[0]["reference_joint_pos"].shape[0], 2)

    def test_legacy_mapping_is_preserved(self):
        hand = SimpleNamespace(body_names=["palm", "a", "b"], to_hand=lambda name: ["wrist" if name == "palm" else "human"])
        env = SimpleNamespace(device="cpu", max_episode_length=10, dexhand_rh=hand)
        human = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        data = {"obj_trajectory": torch.eye(4).repeat(4, 1, 1), "mano_joints": {"human": human}}
        packed = pack_method()(env, [data, data])
        torch.testing.assert_close(packed["mano_joints"][0], torch.cat([human, human], dim=-1))
        with self.assertRaises(ValueError):
            pack_method()(env, [data, dict(data, reference_source="spider")])


if __name__ == "__main__":
    unittest.main()
