"""CPU regressions for the retarget viewer's actual CLI and scene/state hooks."""

import argparse
import ast
from contextlib import redirect_stderr
from io import StringIO
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lib.utils.visualization import VisualizationConfig


ROOT = Path(__file__).resolve().parents[1]
KEYS = ("hand_rh", "hand_lh", "object_rh", "object_lh")
NAMES = dict(zip(KEYS, ("dexhand_r", "dexhand_l", "manip_obj_rh", "manip_obj_lh")))
GYMAPI = SimpleNamespace(
    AssetOptions=SimpleNamespace, Transform=SimpleNamespace,
    Vec3=lambda *args: tuple(float(value) for value in args),
    DOMAIN_SIM="sim", MESH_VISUAL="visual", INVALID_HANDLE=-1,
)


def sdk_arguments(description, headless, custom_parameters):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument("--headless", action="store_true")
    for option in custom_parameters:
        # Match gymutil's supported keys instead of accepting extra argparse
        # capabilities (e.g. nargs) that the installed SDK does not forward.
        kwargs = {key: value for key, value in option.items() if key in ("type", "default", "action", "help")}
        parser.add_argument(option["name"], **kwargs)
    return parser.parse_args()


def script_hooks():
    path = ROOT / "scripts/visualize_taco_retarget.py"
    tree = ast.parse(path.read_text())
    selected = []
    constants = {"DEFAULT_REPLAY_COLORS", "TABLE_WIDTH_OFFSET", "TABLE_HEIGHT", "TABLE_POSITION"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("_parse_rgb", "_parse_bool", "parse_args"):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "TacoRetargetViewer":
            selected.append(node)
    namespace = {
        "argparse": argparse, "VisualizationConfig": VisualizationConfig, "np": np, "torch": torch,
        "gymapi": GYMAPI, "gymtorch": SimpleNamespace(wrap_tensor=lambda value: value, unwrap_tensor=lambda value: value),
        "gymutil": SimpleNamespace(parse_arguments=sdk_arguments),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


HOOKS = script_hooks()
Viewer = HOOKS["TacoRetargetViewer"]


class FakeGym:
    def __init__(self):
        self.actors = []
        self.scales = []
        self.colors = []
        self.textures = []
        self.created_textures = []
        self.root_writes = 0
        self.dof_writes = 0
        self.dof_acquisitions = 0

    def create_box(self, *args):
        return SimpleNamespace(dof_names=[])

    def create_env(self, *args):
        return "env"

    def create_actor(self, env, asset, pose, name, group, collision_filter):
        self.actors.append((name, asset.dof_names))
        return len(self.actors) - 1

    def set_actor_scale(self, env, actor, scale):
        self.scales.append((actor, scale))

    def create_texture_from_buffer(self, sim, width, height, pixels):
        self.created_textures.append((width, height, pixels.copy()))
        return 123

    def get_actor_rigid_body_count(self, env, actor):
        return 3

    def set_rigid_body_color(self, env, actor, index, mesh, color):
        self.colors.append((actor, index, color))

    def set_rigid_body_texture(self, env, actor, index, mesh, texture):
        self.textures.append((actor, index, texture))

    def acquire_actor_root_state_tensor(self, sim):
        self.root = torch.full((len(self.actors), 13), -7.0)
        return self.root

    def get_actor_index(self, env, actor, domain):
        return actor

    def get_sim_dof_count(self, sim):
        return sum(len(names) for _, names in self.actors)

    def acquire_dof_state_tensor(self, sim):
        assert self.get_sim_dof_count(sim) > 0, "An object-only scene must not acquire a DOF tensor"
        self.dof_acquisitions += 1
        self.dofs = torch.full((self.get_sim_dof_count(sim), 2), -7.0)
        return self.dofs

    def find_actor_dof_index(self, env, actor, name, domain):
        offset = sum(len(names) for _, names in self.actors[:actor])
        # Deliberately reverse the state-buffer order to detect fixed slices
        # or an assumption that the left hand starts after the right hand.
        names = self.actors[actor][1]
        return offset + len(names) - 1 - names.index(name)

    def set_actor_root_state_tensor(self, sim, tensor):
        self.root_writes += 1

    def set_dof_state_tensor(self, sim, tensor):
        self.dof_writes += 1


def viewer(visible=(True, True, True, True), preserve=False, graphics=0):
    result = Viewer.__new__(Viewer)
    result.args = SimpleNamespace(graphics_device_id=graphics)
    result.visualization = VisualizationConfig(
        right_hand_color=(0.35, 0.75, 0.95), left_hand_color=(0.75, 0.45, 0.95),
        tool_color=(1, 0.45, 0.08), object_color=(0.2, 0.7, 0.35), preserve_textures=preserve,
    )
    result._visualization_white_texture = None
    result.visible = dict(zip(KEYS, visible))
    result.gym = FakeGym()
    result.sim = "sim"
    result.device = torch.device("cpu")
    result.dexhand_rh = SimpleNamespace(n_dofs=2, dof_names=["rh0", "rh1"])
    result.dexhand_lh = SimpleNamespace(n_dofs=3, dof_names=["lh0", "lh1", "lh2"])
    result.rh_data = {"obj_urdf_path": "tool.urdf", "object_scale": 2.0}
    result.lh_data = {"obj_urdf_path": "target.urdf", "obj_scale": 3.0}
    result.loaded_assets = []

    def load_hand(hand):
        result.loaded_assets.append(hand)
        return hand

    def load_object(path):
        result.loaded_assets.append(path)
        return SimpleNamespace(dof_names=[])

    result._load_hand_asset = load_hand
    result._load_object_asset = load_object
    for index, key in enumerate(KEYS):
        setattr(result, key + "_pos", torch.arange(6.0).reshape(2, 3) + index * 100)
        setattr(result, key + "_quat", torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float32))
    result.hand_rh_dof = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    result.hand_lh_dof = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32)
    return result


class ReplayArgumentTests(unittest.TestCase):
    def test_defaults_keep_all_actors_and_existing_rgb_values(self):
        with patch.object(sys, "argv", ["replay", "--data_idx", "sequence"]):
            args = HOOKS["parse_args"]()
        for role, rgb in HOOKS["DEFAULT_REPLAY_COLORS"].items():
            self.assertEqual(getattr(args, role + "_color"), rgb)
            self.assertTrue(getattr(args, "show_" + role))
        self.assertFalse(args.preserve_textures)
        self.assertEqual(args.reference_source, "legacy")
        self.assertEqual(args.object_reference, "spider_rollout")

    def test_explicit_colors_and_true_false_are_forwarded_by_sdk_parser(self):
        arguments = ["replay", "--right_hand_color", "0.1,0.2,0.3", "--left_hand_color", "0.9,0.8,0.7",
                     "--object_color", "0,1,0", "--tool_color", "1,0,0", "--show_right_hand", "TRUE",
                     "--show_left_hand", "false", "--show_object", "false", "--show_tool", "true",
                     "--preserve_textures", "true", "--reference_source", "spider"]
        with patch.object(sys, "argv", arguments):
            args = HOOKS["parse_args"]()
        self.assertTrue(args.show_right_hand)
        self.assertFalse(args.show_left_hand)
        self.assertFalse(args.show_object)
        self.assertTrue(args.show_tool)
        self.assertTrue(args.preserve_textures)
        self.assertEqual(args.right_hand_color, (0.1, 0.2, 0.3))
        self.assertEqual(args.left_hand_color, (0.9, 0.8, 0.7))
        self.assertEqual(args.object_color, (0, 1, 0))
        self.assertEqual(args.tool_color, (1, 0, 0))

    def test_invalid_colors_and_booleans_are_cli_errors(self):
        for option, values in (("--tool_color", ["red", "0,1", "0,1,2", "0,nan,0", "inf,0,0"]),
                               ("--show_left_hand", ["0", "yes", ""]),
                               ("--preserve_textures", ["1", "no"])):
            for value in values:
                with self.subTest(option=option, value=value), patch.object(sys, "argv", ["replay", option, value]):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                        HOOKS["parse_args"]()
                    self.assertEqual(error.exception.code, 2)


class ReplaySceneTests(unittest.TestCase):
    def test_all_16_visibility_combinations_create_color_scale_and_update_correct_actors(self):
        for visible in product((False, True), repeat=4):
            with self.subTest(visible=visible):
                replay = viewer(visible)
                original = {key: getattr(replay, key + "_pos").clone() for key in KEYS}
                replay._create_scene()
                expected_names = (["dexhand_r"] if visible[0] else []) + (["dexhand_l"] if visible[1] else [])
                expected_names += ["table"] + (["manip_obj_rh"] if visible[2] else []) + (["manip_obj_lh"] if visible[3] else [])
                self.assertEqual([name for name, _ in replay.gym.actors], expected_names)
                self.assertEqual(set(replay.actors), {key for key, enabled in zip(KEYS, visible) if enabled})
                self.assertEqual(len(replay.loaded_assets), sum(visible))
                expected_scales = [(replay.actors[key], scale) for key, scale in (("object_rh", 2), ("object_lh", 3))
                                   if key in replay.actors]
                self.assertEqual(replay.gym.scales, expected_scales)

                expected_colors = dict(zip(KEYS, ((0.35, 0.75, 0.95), (0.75, 0.45, 0.95), (1, 0.45, 0.08), (0.2, 0.7, 0.35))))
                colors = {(actor, body): color for actor, body, color in replay.gym.colors}
                for key, actor in replay.actors.items():
                    for body in range(3):
                        self.assertEqual(colors[actor, body], expected_colors[key])
                self.assertEqual(len(replay.gym.created_textures), 1 if any(visible) else 0)

                replay._acquire_state_tensors()
                replay._set_frame(1)
                self.assertEqual(replay.gym.root_writes, 1)
                self.assertEqual(replay.gym.dof_acquisitions, 1 if any(visible[:2]) else 0)
                self.assertEqual(replay.gym.dof_writes, 1 if any(visible[:2]) else 0)
                for key, actor in replay.actors.items():
                    self.assertTrue(torch.equal(replay.root_state[actor, :3], getattr(replay, key + "_pos")[1]))
                    self.assertTrue(torch.equal(replay.root_state[actor, 3:7], getattr(replay, key + "_quat")[1]))
                    self.assertTrue(torch.all(replay.root_state[actor, 7:] == 0))
                table_index = expected_names.index("table")
                self.assertTrue(torch.all(replay.root_state[table_index] == -7))
                if replay.dof_state is not None:
                    for key, hand in (("hand_rh", replay.dexhand_rh), ("hand_lh", replay.dexhand_lh)):
                        if key in replay.actors:
                            for local, name in enumerate(hand.dof_names):
                                index = replay.gym.find_actor_dof_index("env", replay.actors[key], name, "sim")
                                self.assertEqual(replay.dof_state[index, 0], getattr(replay, key + "_dof")[1, local])
                    self.assertTrue(torch.all(replay.dof_state[:, 1] == 0))
                else:
                    self.assertFalse(any(visible[:2]))
                for key in KEYS:
                    self.assertTrue(torch.equal(original[key], getattr(replay, key + "_pos")))

    def test_camera_uses_only_visible_trajectories_and_table_when_all_hidden(self):
        for visible in product((False, True), repeat=4):
            with self.subTest(visible=visible):
                replay = viewer(visible)
                replay._create_scene()
                positions = [getattr(replay, key + "_pos") for key, enabled in zip(KEYS, visible) if enabled]
                if positions:
                    array = torch.cat(positions).numpy()
                    expected = (array.min(axis=0) + array.max(axis=0)) / 2
                else:
                    expected = np.array(HOOKS["TABLE_POSITION"])
                np.testing.assert_allclose(replay._camera_center(), expected)

    def test_preserve_textures_and_headless_do_not_allocate_neutral_texture(self):
        for preserve, graphics in ((True, 0), (False, -1)):
            replay = viewer(preserve=preserve, graphics=graphics)
            replay._create_scene()
            self.assertEqual(replay.gym.created_textures, [])
            self.assertEqual(replay.gym.textures, [])

    def test_invalid_dof_mapping_fails_instead_of_corrupting_another_hand(self):
        replay = viewer()
        replay._create_scene()
        replay.gym.find_actor_dof_index = lambda *args: -1
        with self.assertRaisesRegex(RuntimeError, "Invalid simulation DOF indices"):
            replay._acquire_state_tensors()

    def test_unexpected_articulated_object_dofs_are_rejected(self):
        replay = viewer((False, False, True, False))
        replay._create_scene()
        replay.gym.get_sim_dof_count = lambda *args: 1
        with self.assertRaisesRegex(RuntimeError, "Unexpected simulation DOF count"):
            replay._acquire_state_tensors()


if __name__ == "__main__":
    unittest.main()
