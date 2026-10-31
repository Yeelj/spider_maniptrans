"""CPU regressions for the real bimanual visualization hooks, without Isaac Gym."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from lib.utils.visualization import VisualizationConfig


ROOT = Path(__file__).resolve().parents[1]
GYMAPI = SimpleNamespace(
    MESH_VISUAL="visual", INVALID_HANDLE=-1, DOMAIN_SIM="sim",
    Vec3=lambda *channels: tuple(float(channel) for channel in channels),
    Quat=lambda *channels: tuple(channels),
    Transform=SimpleNamespace, AssetOptions=SimpleNamespace,
)


def task_class():
    path = ROOT / "maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DexHandManipBiHEnv")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "VecTask": object, "VisualizationConfig": VisualizationConfig,
        "gymapi": GYMAPI, "torch": torch, "np": np,
        "rotmat_to_quat": lambda rotation: torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[cls.name]


class FakeGym:
    def __init__(self):
        self.colors = []
        self.textures = []
        self.actors = []
        self.lines = []
        self.cleared = []
        self.spheres = []
        self.created_textures = []

    def create_texture_from_buffer(self, sim, width, height, pixels):
        self.created_textures.append((sim, width, height, pixels.copy()))
        return 99

    def get_actor_rigid_body_count(self, env, actor):
        return 3

    def set_rigid_body_color(self, env, actor, body, mesh, color):
        self.colors.append((env, actor, body, mesh, color))

    def set_rigid_body_texture(self, env, actor, body, mesh, texture):
        self.textures.append((env, actor, body, mesh, texture))

    def create_actor(self, env, asset, pose, name, group, collision_filter):
        self.actors.append(name)
        return name

    def get_actor_index(self, env, actor, domain):
        return self.actors.index(actor)

    def find_actor_handle(self, env, name):
        return name

    def create_sphere(self, sim, radius, options):
        self.spheres.append(radius)
        return "sphere"

    def clear_lines(self, viewer):
        self.cleared.append(viewer)

    def add_lines(self, viewer, env, count, line, color):
        self.lines.append((env, line.copy(), color.copy()))


ENV = task_class()


def environment(options=None):
    env = ENV.__new__(ENV)
    env.gym = FakeGym()
    env.visualization = VisualizationConfig.from_mapping(options)
    env.graphics_device_id = 0
    env._visualization_white_texture = None
    env.headless = False
    env.show_reference_skeleton = env.visualization.show_reference_skeleton
    env.envs = ["env0", "env1"]
    env.num_envs = 2
    env.sim = "sim"
    env.viewer = "viewer"
    return env


class VisualizationConfigTests(unittest.TestCase):
    def test_missing_options_preserve_legacy_defaults(self):
        self.assertEqual(VisualizationConfig.from_mapping(), VisualizationConfig())
        self.assertEqual(VisualizationConfig.from_mapping({}), VisualizationConfig())
        self.assertTrue(VisualizationConfig().show_reference_skeleton)
        self.assertTrue(VisualizationConfig().show_contact_highlight)

    def test_invalid_options_fail_before_simulation(self):
        for value in ([], False, {"typo": True}, {"showContactHighlight": "false"},
                      {"showReferenceSkeleton": 0}, {"preserveTextures": None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VisualizationConfig.from_mapping(value)
        for name in ("rightHandColor", "leftHandColor", "objectColor", "toolColor"):
            for value in ([0, 1], [0, 0, 0, 1], "red", [0, 1, -0.1], [1.1, 0, 0],
                          [0, float("nan"), 0], [0, float("inf"), 0], [True, 0, 0], ["1", 0, 0]):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    VisualizationConfig.from_mapping({name: value})

    def test_rgb_is_copied_without_mutating_config(self):
        rgb = [0, 0.5, 1]
        config = VisualizationConfig.from_mapping({"rightHandColor": rgb, "leftHandColor": [1, 0, 0]})
        rgb[0] = 1
        self.assertEqual(config.hand_color("rh"), (0.0, 0.5, 1.0))
        self.assertEqual(config.hand_color("lh"), (1.0, 0.0, 0.0))

    def test_hydra_accepts_test_command_overrides_without_plus(self):
        with initialize_config_dir(config_dir=str(ROOT / "main/cfg"), version_base="1.1"):
            cfg = compose(config_name="config", overrides=[
                "task=ResDexHand", "side=BiH", "test=true",
                "task.env.visualization.rightHandColor=[0.35,0.75,0.95]",
                "task.env.visualization.leftHandColor=[0.75,0.45,0.95]",
                "task.env.visualization.objectColor=[0.2,0.7,0.35]",
                "task.env.visualization.toolColor=[1,0.45,0.08]",
                "task.env.visualization.showReferenceSkeleton=false",
                "task.env.visualization.showContactHighlight=false",
            ])
            config = VisualizationConfig.from_mapping(OmegaConf.to_container(cfg.task.env.visualization))
            self.assertFalse(config.show_reference_skeleton)
            self.assertFalse(config.show_contact_highlight)
            self.assertEqual(config.tool_color, (1.0, 0.45, 0.08))
            self.assertFalse(cfg.task.env.training)
            self.assertFalse(cfg.task.task.randomize)


class VisualizationHookTests(unittest.TestCase):
    def test_null_color_does_not_touch_original_materials(self):
        env = environment()
        env._set_actor_color("env0", "hand", None)
        self.assertEqual(env.gym.colors, [])
        self.assertEqual(env.gym.textures, [])

    def test_custom_color_covers_all_actor_bodies_and_optionally_keeps_texture(self):
        for preserve in (False, True):
            env = environment({"preserveTextures": preserve})
            env._set_actor_color("env0", "hand", (0.1, 0.2, 0.3))
            self.assertEqual([call[2] for call in env.gym.colors], [0, 1, 2])
            self.assertTrue(all(call[3:] == ("visual", (0.1, 0.2, 0.3)) for call in env.gym.colors))
            self.assertEqual(len(env.gym.textures), 0 if preserve else 3)
            self.assertTrue(all(call[-1] == 99 for call in env.gym.textures))
            self.assertEqual(len(env.gym.created_textures), 0 if preserve else 1)
            if not preserve:
                sim, width, height, pixels = env.gym.created_textures[0]
                self.assertEqual((sim, width, height), ("sim", 1, 1))
                np.testing.assert_array_equal(pixels, np.full((1, 4), 255, dtype=np.uint8))
                env._set_actor_color("env1", "other_hand", (0.3, 0.2, 0.1))
                self.assertEqual(len(env.gym.created_textures), 1)

    def test_no_graphics_does_not_allocate_textures_and_texture_failure_is_explicit(self):
        env = environment()
        env.graphics_device_id = -1
        env._set_actor_color("env0", "hand", (0.1, 0.2, 0.3))
        self.assertEqual(env.gym.created_textures, [])
        self.assertEqual(env.gym.colors, [])
        env.graphics_device_id = 0
        env.gym.create_texture_from_buffer = lambda *args: -1
        with self.assertRaisesRegex(RuntimeError, "neutral visualization texture"):
            env._set_actor_color("env0", "hand", (0.1, 0.2, 0.3))

    def test_tool_and_target_colors_are_independent_and_disabled_skeleton_creates_no_spheres(self):
        env = environment({"showReferenceSkeleton": False, "toolColor": [1, 0, 0], "objectColor": [0, 1, 0]})
        for side in ("rh", "lh"):
            setattr(env, "demo_data_" + side, {"obj_trajectory": torch.eye(4)[None, None], "scene_objs": [[]]})
            actor, index = env._create_obj_actor("env0", 0, "object_asset", side)
            self.assertEqual(actor, "manip_obj_" + side)
            self.assertEqual(index, 0 if side == "rh" else 1)
        self.assertEqual(env.gym.actors, ["manip_obj_rh", "manip_obj_lh"])
        self.assertEqual(env.gym.spheres, [])
        self.assertEqual([call[-1] for call in env.gym.colors], [(1, 0, 0)] * 3 + [(0, 1, 0)] * 3)

    def test_enabled_skeleton_still_creates_both_sides_joint_markers(self):
        env = environment()
        for side in ("rh", "lh"):
            setattr(env, "dexhand_" + side, SimpleNamespace(body_names=["palm", "index_tip"], to_hand=lambda name: [name]))
            setattr(env, "demo_data_" + side, {"obj_trajectory": torch.eye(4)[None, None], "scene_objs": [[]]})
            env._create_obj_actor("env0", 0, "object_asset", side)
        self.assertEqual(env.gym.actors, ["manip_obj_rh", "rh_mano_joint_0", "rh_mano_joint_1",
                                          "manip_obj_lh", "lh_mano_joint_0", "lh_mano_joint_1"])
        self.assertEqual(len(env.gym.spheres), 4)
        self.assertEqual(env.gym.textures, [])

    def test_headless_and_disabled_visuals_do_not_access_skeleton_or_contacts(self):
        env = environment()
        env.headless = True
        env._update_visualization()
        env.headless = False
        env.show_reference_skeleton = False
        env.visualization = VisualizationConfig(show_contact_highlight=False)
        env._update_visualization()
        self.assertEqual(env.gym.colors, [])
        self.assertEqual(env.gym.lines, [])
        self.assertEqual(env.gym.cleared, [])

    def test_contacts_use_each_actor_local_index_and_restore_custom_base_colors(self):
        env = environment({"showReferenceSkeleton": False, "rightHandColor": [0.2, 0.3, 0.4],
                           "leftHandColor": [0.7, 0.8, 0.9]})
        env.net_cf = torch.zeros(2, 12, 3)
        env.net_cf[0, 8, 2] = 1
        env._hand_visual_body_indices = {"rh": {"palm": 1, "tip": 0}, "lh": {"palm": 0, "tip": 1}}
        for side, offset in (("rh", 3), ("lh", 7)):
            setattr(env, "dexhand_" + side, SimpleNamespace(body_names=["palm", "tip"]))
            setattr(env, "dexhand_" + side + "_handles", {"palm": offset, "tip": offset + 1})
        before = env.net_cf.clone()
        env._update_visualization()
        colors = {(env_id, actor, body): color for env_id, actor, body, _, color in env.gym.colors}
        self.assertEqual(colors["env0", "dexhand_l", 1], (1.0, 0.6, 0.6))
        self.assertEqual(colors["env0", "dexhand_l", 0], (0.7, 0.8, 0.9))
        self.assertEqual(colors["env1", "dexhand_l", 1], (0.7, 0.8, 0.9))
        self.assertEqual(colors["env0", "dexhand_r", 1], (0.2, 0.3, 0.4))
        self.assertTrue(torch.equal(before, env.net_cf))
        self.assertEqual(env.gym.lines, [])

    def test_legacy_contact_colors_remain_pink_and_white(self):
        env = environment()
        env._hand_visual_body_indices = {"rh": {"palm": 0}}
        env.set_force_vis("env0", "palm", True, "rh")
        env.set_force_vis("env0", "palm", False, "rh")
        self.assertEqual([call[-1] for call in env.gym.colors], [(1.0, 0.6, 0.6), (1.0, 1.0, 1.0)])

    def test_skeleton_without_highlights_clamps_frames_and_preserves_reference_data(self):
        env = environment({"showContactHighlight": False})
        env.progress_buf = torch.tensor([1, 5])
        for side, bodies in (("rh", 3), ("lh", 2)):
            setattr(env, "dexhand_" + side, SimpleNamespace(n_bodies=bodies, bone_links=[(0, j) for j in range(1, bodies)]))
            demo = {"seq_len": torch.tensor([3, 2]), "wrist_pos": torch.arange(18.0).reshape(2, 3, 3),
                    "mano_joints": torch.arange(2 * 3 * (bodies - 1) * 3.0).reshape(2, 3, -1)}
            setattr(env, "demo_data_" + side, demo)
            setattr(env, "mano_joint_" + side + "_points", [torch.full((2, 13), -1.0) for _ in range(bodies)])
        before = {side: {key: value.clone() for key, value in getattr(env, "demo_data_" + side).items()}
                  for side in ("rh", "lh")}
        env._update_visualization()
        self.assertEqual(env.gym.cleared, ["viewer"])
        self.assertEqual(len(env.gym.lines), 6)
        self.assertEqual(env.gym.colors, [])
        for side in ("rh", "lh"):
            for key, value in before[side].items():
                self.assertTrue(torch.equal(value, getattr(env, "demo_data_" + side)[key]))
            points = getattr(env, "mano_joint_" + side + "_points")
            self.assertTrue(torch.equal(points[0][:, :3], before[side]["wrist_pos"][:, 1]))
            self.assertTrue(torch.equal(points[1][:, :3], before[side]["mano_joints"][:, 1, :3]))
            self.assertTrue(all(torch.all(point[:, 3:] == -1) for point in points))


if __name__ == "__main__":
    unittest.main()
