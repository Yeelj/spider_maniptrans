#!/usr/bin/env python3
"""Replay bimanual legacy or SPIDER retargeting results in Isaac Gym.

The retargeting pickle contains only dexterous-hand states.  Object assets,
object trajectories, and the source-to-Gym transform are intentionally loaded
through the existing dataset loaders. ``--new_obj`` selects a replacement
right-hand object from the corresponding dataset, retaining the original hand
and object trajectories. Use ``--reference_source spider`` to load the same
validated SPIDER bundles and assets as training instead.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Isaac Gym must be imported before torch.
try:
    from isaacgym import gymapi, gymtorch, gymutil
except ModuleNotFoundError as error:
    if error.name != "isaacgym":
        raise
    raise SystemExit(
        f"Isaac Gym is not visible to {sys.executable}. "
        "Use the Python interpreter from the environment where Isaac Gym is installed; "
        "see docs/retarget_visualization.md for physgraph launch commands."
    ) from error

import numpy as np
import torch

from main.dataset.factory import ManipDataFactory
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_object_paths as _oakink2_object_paths
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from lib.utils.visualization import VisualizationConfig


TABLE_WIDTH_OFFSET = 0.2
TABLE_HEIGHT = 0.03
TABLE_POSITION = (-TABLE_WIDTH_OFFSET / 2, 0.0, 0.4)
TABLE_SURFACE_Z = TABLE_POSITION[2] + TABLE_HEIGHT / 2
REQUIRED_RETARGET_KEYS = ("opt_wrist_pos", "opt_wrist_rot", "opt_dof_pos")
DEFAULT_REPLAY_COLORS = {
    "right_hand": (0.35, 0.75, 0.95),
    "left_hand": (0.75, 0.45, 0.95),
    "object": (0.2, 0.7, 0.35),
    "tool": (1.0, 0.45, 0.08),
}


def _parse_rgb(value):
    try:
        channels = [float(channel) for channel in value.split(",")]
        return VisualizationConfig.from_mapping({"objectColor": channels}).object_color
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected three finite RGB values in [0,1], e.g. 0.35,0.75,0.95") from error


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected true or false")


def _oakink2_annotation_path(data_dir: Path, data_idx: str) -> Path:
    sequence_hash, separator, stage_text = data_idx.partition("@")
    if not separator or len(sequence_hash) != 5 or not stage_text.isdigit():
        raise ValueError(f"Expected an OakInk-V2 sequence ID such as '9b728@0', got {data_idx!r}")

    annotation_dir = data_dir / "anno_preview"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing OakInk-V2 annotation directory: {annotation_dir}")
    # Match the same five-character hash used by OakInk2DatasetDexHand, but
    # reject collisions instead of silently choosing a different sequence.
    matches = []
    for path in sorted(annotation_dir.glob("*.pkl")):
        fields = path.stem.split("_")
        if len(fields) > 5 and fields[5][:5] == sequence_hash:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No OakInk-V2 annotation for {data_idx!r} in {annotation_dir}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous OakInk-V2 sequence hash {sequence_hash!r}: {matches}")
    return matches[0]


def _retarget_path(dexhand, data_idx: str, dataset_type="taco", annotation_path=None) -> Path:
    if dataset_type == "taco":
        dataset_dir = "TACO"
        filename = f"{data_idx}.pkl"
    elif dataset_type == "oakink2":
        dataset_dir = "OakInk-v2"
        stage = int(data_idx.partition("@")[2])
        filename = f"{annotation_path.stem}@{stage}.pkl"
    else:
        raise ValueError(f"Unsupported replay dataset: {dataset_type!r}")
    return PROJECT_ROOT / "data" / "retargeting" / dataset_dir / f"mano2{dexhand}" / filename


def _set_oakink2_object_asset(data, data_dir: Path, object_id: str):
    mesh_path, urdf_path = _oakink2_object_paths(data_dir, object_id)
    # These fields select the viewer asset only. obj_trajectory and all hand
    # states remain in the original retargeting frame.
    data["obj_id"] = object_id
    data["obj_mesh_path"] = mesh_path
    data["obj_urdf_path"] = urdf_path
    data["obj_scale"] = oakink2_obj_scale.get(object_id, 1.0)


def _load_and_validate_pickle(path: Path, expected_dofs: int):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing retargeting result: {path}. Run main/dataset/mano2dexhand.py "
            "for this sequence, hand side, and dexhand first."
        )

    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(data).__name__}")

    missing = [key for key in REQUIRED_RETARGET_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    expected_tail_shapes = {
        "opt_wrist_pos": (3,),
        "opt_wrist_rot": (3,),
        "opt_dof_pos": (expected_dofs,),
    }
    frame_count = None
    for key, tail_shape in expected_tail_shapes.items():
        value = np.asarray(data[key])
        if value.ndim != 2 or value.shape[1:] != tail_shape:
            raise ValueError(
                f"{path}: {key} must have shape [T, {', '.join(map(str, tail_shape))}], "
                f"got {value.shape}"
            )
        if value.shape[0] < 1:
            raise ValueError(f"{path}: {key} contains no frames")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains NaN or Inf")
        if frame_count is None:
            frame_count = value.shape[0]
        elif value.shape[0] != frame_count:
            raise ValueError(
                f"{path}: inconsistent frame counts; expected {frame_count}, "
                f"but {key} has {value.shape[0]}"
            )
    return data, frame_count


def _mujoco_to_gym_transform(device: torch.device) -> torch.Tensor:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2], dtype=np.float32)) @ aa_to_rotmat(
        np.array([np.pi / 2, 0, 0], dtype=np.float32)
    )
    transform[:3, 3] = np.array([0, 0, TABLE_SURFACE_Z], dtype=np.float32)
    return torch.as_tensor(transform, dtype=torch.float32, device=device)


def _validate_loaded_data(data, side: str, frame_count: int, expected_dofs: int) -> None:
    expected_shapes = {
        "opt_wrist_pos": (frame_count, 3),
        "opt_wrist_rot": (frame_count, 3),
        "opt_dof_pos": (frame_count, expected_dofs),
        "obj_trajectory": (frame_count, 4, 4),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in data:
            raise KeyError(f"The {side}-hand sample is missing {key!r}")
        if tuple(data[key].shape) != expected_shape:
            raise ValueError(
                f"The {side}-hand sample has {key}.shape={tuple(data[key].shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(data[key]).all().item():
            raise ValueError(f"The {side}-hand sample has NaN or Inf in {key}")

    urdf_path = Path(data["obj_urdf_path"])
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Missing {side}-hand object URDF: {urdf_path}")


def _load_spider_replay_data(args, device, dexhand_rh, dexhand_lh, dataset_dir):
    from lib.utils.spider_reference import find_manifest, read_bundle

    reference_root = Path(args.reference_root)
    if not reference_root.is_absolute():
        reference_root = PROJECT_ROOT / reference_root
    manifest_path = find_manifest(reference_root, args.new_obj or None, dataset_dir, args.data_idx)
    manifest, _ = read_bundle(manifest_path)
    frame_count = manifest["recorded_frames"]
    samples = []
    for side, hand in (("right", dexhand_rh), ("left", dexhand_lh)):
        dataset = ManipDataFactory.create_data(
            manipdata_type="spider",
            side=side,
            dexhand=hand,
            reference_root=str(reference_root),
            reference_dataset=dataset_dir,
            new_obj=args.new_obj or None,
            object_reference=args.object_reference,
            control_dt=1 / manifest["fps"],
            device=device,
        )
        data = dataset[args.data_idx]
        _validate_loaded_data(data, side, frame_count, hand.n_dofs)
        samples.append(data)

    print(f"SPIDER manifest: {manifest_path}")
    print(f"Frames: {frame_count} at {manifest['fps']:g} Hz; playback: {args.playback_fps:g} FPS")
    print(f"Object reference: {args.object_reference}; validation: {manifest['validation_status']}")
    for side, data in zip(("Right", "Left"), samples):
        print(f"{side} retarget: {data['data_path']}")
        print(f"{side} object: {data['selected_asset_id']} ({data['obj_urdf_path']})")
    return samples[0], samples[1], frame_count


@torch.no_grad()
def _load_replay_data(args, device, dexhand_rh, dexhand_lh):
    if not args.data_idx:
        raise ValueError("--data_idx is required")
    dataset_type = ManipDataFactory.dataset_type(args.data_idx)
    if dataset_type not in ("taco", "oakink2"):
        raise ValueError(
            f"Expected a TACO ID such as 'db_teapot_0930049' or an OakInk-V2 ID such as '9b728@0', "
            f"got {args.data_idx!r}"
        )

    dataset_dir = "TACO" if dataset_type == "taco" else "OakInk-v2"
    if args.reference_source == "spider":
        return _load_spider_replay_data(args, device, dexhand_rh, dexhand_lh, dataset_dir)

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / dataset_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    annotation_path = _oakink2_annotation_path(data_dir, args.data_idx) if dataset_type == "oakink2" else None
    rh_pickle = _retarget_path(dexhand_rh, args.data_idx, dataset_type, annotation_path)
    lh_pickle = _retarget_path(dexhand_lh, args.data_idx, dataset_type, annotation_path)
    rh_retarget, rh_frames = _load_and_validate_pickle(rh_pickle, dexhand_rh.n_dofs)
    lh_retarget, lh_frames = _load_and_validate_pickle(lh_pickle, dexhand_lh.n_dofs)
    if rh_frames != lh_frames:
        raise ValueError(f"Right/left retarget frame mismatch: {rh_frames} vs {lh_frames}")

    transform = _mujoco_to_gym_transform(device)
    common_kwargs = {
        "data_dir": str(data_dir),
        "device": device,
        "mujoco2gym_transf": transform,
        "verbose": False,
    }
    rh_kwargs = {"new_obj": args.new_obj or None}
    rh_dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side="right",
        dexhand=dexhand_rh,
        **common_kwargs,
        **rh_kwargs,
    )
    lh_dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side="left",
        dexhand=dexhand_lh,
        **common_kwargs,
    )
    rh_data = rh_dataset[args.data_idx]
    lh_data = lh_dataset[args.data_idx]
    # Use the already validated files directly. This also avoids depending on
    # the caller's current working directory; the dataset's legacy PKL lookup
    # is relative to the repository root.
    for key in REQUIRED_RETARGET_KEYS:
        rh_data[key] = torch.as_tensor(rh_retarget[key], dtype=torch.float32, device=device)
        lh_data[key] = torch.as_tensor(lh_retarget[key], dtype=torch.float32, device=device)
    if dataset_type == "oakink2":
        _set_oakink2_object_asset(rh_data, data_dir, args.new_obj or rh_data["obj_id"])
        _set_oakink2_object_asset(lh_data, data_dir, lh_data["obj_id"])
    _validate_loaded_data(rh_data, "right", rh_frames, dexhand_rh.n_dofs)
    _validate_loaded_data(lh_data, "left", lh_frames, dexhand_lh.n_dofs)

    print(f"Dataset: {dataset_dir}; sequence: {args.data_idx}")
    print(f"Frames: {rh_frames} at {args.playback_fps:g} FPS")
    print(f"Right retarget: {rh_pickle}")
    print(f"Left retarget:  {lh_pickle}")
    print(f"Right tool: {rh_data['obj_id']} ({rh_data['obj_urdf_path']})")
    print(f"Left target: {lh_data['obj_id']} ({lh_data['obj_urdf_path']})")
    return rh_data, lh_data, rh_frames


class TacoRetargetViewer:
    def __init__(self, args):
        if args.physics_engine != gymapi.SIM_PHYSX:
            raise ValueError("This viewer supports the PhysX backend only")
        if args.playback_fps <= 0:
            raise ValueError(f"playback_fps must be positive, got {args.playback_fps}")
        if args.start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {args.start_frame}")

        self.args = args
        self.visualization = VisualizationConfig.from_mapping({
            "rightHandColor": getattr(args, "right_hand_color", DEFAULT_REPLAY_COLORS["right_hand"]),
            "leftHandColor": getattr(args, "left_hand_color", DEFAULT_REPLAY_COLORS["left_hand"]),
            "objectColor": getattr(args, "object_color", DEFAULT_REPLAY_COLORS["object"]),
            "toolColor": getattr(args, "tool_color", DEFAULT_REPLAY_COLORS["tool"]),
            "preserveTextures": getattr(args, "preserve_textures", False),
        })
        self.visible = {
            "hand_rh": getattr(args, "show_right_hand", True),
            "hand_lh": getattr(args, "show_left_hand", True),
            "object_rh": getattr(args, "show_tool", True),
            "object_lh": getattr(args, "show_object", True),
        }
        if any(not isinstance(value, bool) for value in self.visible.values()):
            raise ValueError("Replay visibility options must be true or false")
        self._visualization_white_texture = None
        if args.headless:
            args.graphics_device_id = -1
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        self.device = torch.device(args.sim_device if args.use_gpu_pipeline else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for device {self.device}, but torch.cuda.is_available() is false")

        try:
            self.dexhand_rh = DexHandFactory.create_hand(args.dexhand, "right")
            self.dexhand_lh = DexHandFactory.create_hand(args.dexhand, "left")
            self.rh_data, self.lh_data, self.frame_count = _load_replay_data(
                args, self.device, self.dexhand_rh, self.dexhand_lh
            )
            if args.start_frame >= self.frame_count:
                raise ValueError(
                    f"start_frame={args.start_frame} is outside a sequence with {self.frame_count} frames"
                )

            self._create_sim()
            self._create_scene()
            self.gym.prepare_sim(self.sim)
            self._acquire_state_tensors()
            self._prepare_trajectories()
            self._create_viewer()
        except Exception:
            self.close()
            raise

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.args.playback_fps
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # Every pose is supplied by the recorded trajectory. Gravity is off so
        # the viewer does not introduce dynamics between consecutive frames.
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu

        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create the Isaac Gym simulation")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_hand_asset(self, dexhand):
        options = gymapi.AssetOptions()
        options.thickness = 0.001
        options.fix_base_link = False
        options.disable_gravity = True
        options.flip_visual_attachments = False
        options.collapse_fixed_joints = False
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        options.use_mesh_materials = True
        urdf_path = os.path.abspath(dexhand.urdf_path)
        asset = self.gym.load_asset(self.sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), options)
        if asset is None:
            raise RuntimeError(f"Failed to load dexhand asset: {urdf_path}")

        asset_dof_names = list(self.gym.get_asset_dof_names(asset))
        if asset_dof_names != list(dexhand.dof_names):
            raise ValueError(
                f"DOF order mismatch for {dexhand}: asset has {asset_dof_names}, "
                f"retarget expects {dexhand.dof_names}"
            )
        return asset

    def _load_object_asset(self, urdf_path):
        options = gymapi.AssetOptions()
        options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        options.thickness = 0.001
        options.fix_base_link = False
        options.disable_gravity = True
        options.collapse_fixed_joints = False
        options.use_mesh_materials = True
        urdf_path = os.path.abspath(urdf_path)
        asset = self.gym.load_asset(self.sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), options)
        if asset is None:
            raise RuntimeError(f"Failed to load object asset: {urdf_path}")
        return asset

    def _create_scene(self):
        hand_rh_asset = self._load_hand_asset(self.dexhand_rh) if self.visible["hand_rh"] else None
        hand_lh_asset = self._load_hand_asset(self.dexhand_lh) if self.visible["hand_lh"] else None
        object_rh_asset = self._load_object_asset(self.rh_data["obj_urdf_path"]) if self.visible["object_rh"] else None
        object_lh_asset = self._load_object_asset(self.lh_data["obj_urdf_path"]) if self.visible["object_lh"] else None

        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, 0.8 + TABLE_WIDTH_OFFSET, 1.6, TABLE_HEIGHT, table_options
        )

        lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
        identity = gymapi.Transform()
        # A shared non-zero filter bit suppresses contacts between replay
        # actors. Their poses should come solely from the recorded trajectory.
        replay_filter = 1
        self.hand_rh_actor = self.gym.create_actor(
            self.env, hand_rh_asset, identity, "dexhand_r", 0, replay_filter
        ) if self.visible["hand_rh"] else None
        self.hand_lh_actor = self.gym.create_actor(
            self.env, hand_lh_asset, identity, "dexhand_l", 0, replay_filter
        ) if self.visible["hand_lh"] else None

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*TABLE_POSITION)
        self.table_actor = self.gym.create_actor(
            self.env, table_asset, table_pose, "table", 0, replay_filter
        )
        self.object_rh_actor = self.gym.create_actor(
            self.env, object_rh_asset, identity, "manip_obj_rh", 0, replay_filter
        ) if self.visible["object_rh"] else None
        self.object_lh_actor = self.gym.create_actor(
            self.env, object_lh_asset, identity, "manip_obj_lh", 0, replay_filter
        ) if self.visible["object_lh"] else None
        self.actors = {
            key: actor for key, actor in (
                ("hand_rh", self.hand_rh_actor), ("hand_lh", self.hand_lh_actor),
                ("object_rh", self.object_rh_actor), ("object_lh", self.object_lh_actor),
            ) if actor is not None
        }
        for actor, data in ((self.object_rh_actor, self.rh_data), (self.object_lh_actor, self.lh_data)):
            scale = data.get("object_scale", data.get("obj_scale", 1.0))
            if actor is not None and scale != 1.0:
                self.gym.set_actor_scale(self.env, actor, scale)

        colors = {
            "hand_rh": self.visualization.right_hand_color, "hand_lh": self.visualization.left_hand_color,
            "object_rh": self.visualization.tool_color, "object_lh": self.visualization.object_color,
        }
        for key, actor in self.actors.items():
            self._set_actor_color(actor, colors[key])
        self.gym.set_rigid_body_color(self.env, self.table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.12, 0.12, 0.12))

    def _set_actor_color(self, actor, color):
        if color is None or self.args.graphics_device_id < 0:
            return
        if not self.visualization.preserve_textures and self._visualization_white_texture is None:
            self._visualization_white_texture = self.gym.create_texture_from_buffer(
                self.sim, 1, 1, np.full((1, 4), 255, dtype=np.uint8)
            )
            if self._visualization_white_texture == gymapi.INVALID_HANDLE:
                raise RuntimeError("Failed to create the neutral visualization texture")
        body_count = self.gym.get_actor_rigid_body_count(self.env, actor)
        for body_index in range(body_count):
            if not self.visualization.preserve_textures:
                self.gym.set_rigid_body_texture(
                    self.env, actor, body_index, gymapi.MESH_VISUAL, self._visualization_white_texture
                )
            self.gym.set_rigid_body_color(self.env, actor, body_index, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

    def _acquire_state_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(root_tensor).view(-1, 13)
        self.actor_indices = {
            key: self.gym.get_actor_index(self.env, actor, gymapi.DOMAIN_SIM) for key, actor in self.actors.items()
        }

        hands = {"hand_rh": self.dexhand_rh, "hand_lh": self.dexhand_lh}
        expected_dofs = sum(hand.n_dofs for key, hand in hands.items() if key in self.actors)
        actual_dofs = self.gym.get_sim_dof_count(self.sim)
        if actual_dofs != expected_dofs:
            raise RuntimeError(
                f"Unexpected simulation DOF count: {actual_dofs}, expected {expected_dofs}"
            )
        self.dof_state = None
        self.hand_dof_indices = {}
        # PhysX has no DOF tensor for an object-only or table-only scene.
        if expected_dofs == 0:
            return
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_tensor).view(-1, 2)
        for key, hand in hands.items():
            if key not in self.actors:
                continue
            indices = [self.gym.find_actor_dof_index(self.env, self.actors[key], name, gymapi.DOMAIN_SIM)
                       for name in hand.dof_names]
            if len(set(indices)) != hand.n_dofs or any(index < 0 or index >= actual_dofs for index in indices):
                raise RuntimeError(f"Invalid simulation DOF indices for {key}: {indices}")
            self.hand_dof_indices[key] = torch.tensor(indices, dtype=torch.long, device=self.device)

    def _prepare_trajectories(self):
        self.hand_rh_pos = self.rh_data["opt_wrist_pos"].to(self.device, dtype=torch.float32)
        self.hand_lh_pos = self.lh_data["opt_wrist_pos"].to(self.device, dtype=torch.float32)
        self.hand_rh_quat = aa_to_quat(self.rh_data["opt_wrist_rot"]).to(self.device)[:, [1, 2, 3, 0]]
        self.hand_lh_quat = aa_to_quat(self.lh_data["opt_wrist_rot"]).to(self.device)[:, [1, 2, 3, 0]]
        self.hand_rh_dof = self.rh_data["opt_dof_pos"].to(self.device, dtype=torch.float32)
        self.hand_lh_dof = self.lh_data["opt_dof_pos"].to(self.device, dtype=torch.float32)

        rh_object = self.rh_data["obj_trajectory"].to(self.device, dtype=torch.float32)
        lh_object = self.lh_data["obj_trajectory"].to(self.device, dtype=torch.float32)
        self.object_rh_pos = rh_object[:, :3, 3]
        self.object_lh_pos = lh_object[:, :3, 3]
        self.object_rh_quat = rotmat_to_quat(rh_object[:, :3, :3])[:, [1, 2, 3, 0]]
        self.object_lh_quat = rotmat_to_quat(lh_object[:, :3, :3])[:, [1, 2, 3, 0]]

    def _camera_center(self):
        positions = [getattr(self, key + "_pos") for key in self.actors]
        if not positions:
            return np.asarray(TABLE_POSITION, dtype=np.float32)
        all_positions = torch.cat(positions, dim=0)
        center = (all_positions.amin(dim=0) + all_positions.amax(dim=0)) / 2
        return center.detach().cpu().numpy()

    def _create_viewer(self):
        if self.args.headless:
            return
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create the Isaac Gym viewer")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "restart")

        center = self._camera_center()
        camera_target = gymapi.Vec3(float(center[0]), float(center[1]), float(center[2]))
        camera_position = gymapi.Vec3(
            float(center[0] + 0.9), float(center[1] + 0.9), float(max(center[2] + 0.65, 0.8))
        )
        self.gym.viewer_camera_look_at(self.viewer, self.env, camera_position, camera_target)
        print("Controls: Space=pause/resume, R=restart, Esc=quit")

    def _write_root_pose(self, key, position, quaternion):
        state = self.root_state[self.actor_indices[key]]
        state[:3] = position
        state[3:7] = quaternion
        state[7:13] = 0

    def _set_frame(self, frame: int):
        for key in self.actors:
            self._write_root_pose(key, getattr(self, key + "_pos")[frame], getattr(self, key + "_quat")[frame])
        if self.dof_state is not None:
            self.dof_state[:, 1] = 0
            for key, indices in self.hand_dof_indices.items():
                self.dof_state[indices, 0] = getattr(self, key + "_dof")[frame]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))
        if self.dof_state is not None:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def run(self):
        frame = self.args.start_frame
        paused = False
        quit_requested = False
        while not quit_requested and (self.viewer is None or not self.gym.query_viewer_has_closed(self.viewer)):
            restart_requested = False
            events = self.gym.query_viewer_action_events(self.viewer) if self.viewer is not None else ()
            for event in events:
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    quit_requested = True
                elif event.action == "pause":
                    paused = not paused
                elif event.action == "restart":
                    restart_requested = True

            if quit_requested:
                break
            if restart_requested:
                frame = self.args.start_frame
                paused = False

            reached_end = False
            if not paused:
                self._set_frame(frame)
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                reached_end = frame + 1 >= self.frame_count

            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

            if not paused:
                if reached_end:
                    if self.args.once or self.args.headless:
                        break
                    frame = self.args.start_frame
                else:
                    frame += 1

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


def parse_args():
    args = gymutil.parse_arguments(
        description="Replay bimanual legacy or SPIDER retargeting results in Isaac Gym",
        headless=True,
        custom_parameters=[
            {
                "name": "--reference_source",
                "type": str,
                "default": "legacy",
                "help": "Trajectory source: original retarget pickles or validated SPIDER training bundles",
            },
            {
                "name": "--reference_root",
                "type": str,
                "default": "data/retargeting_spider_native",
                "help": "SPIDER bundle root, relative to the project root unless absolute",
            },
            {
                "name": "--object_reference",
                "type": str,
                "default": "spider_rollout",
                "help": "SPIDER object poses: actual physical replay (default) or aligned demonstration targets",
            },
            {
                "name": "--data_idx",
                "type": str,
                "help": "TACO sequence ID (e.g. db_teapot_0930049) or OakInk-V2 hash@stage (e.g. 9b728@0)",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "artimano",
                "help": "Dexhand embodiment used to create the retarget pickle",
            },
            {
                "name": "--data_dir",
                "type": str,
                "default": "",
                "help": "Dataset root; defaults to data/TACO or data/OakInk-v2 according to data_idx",
            },
            {
                "name": "--new_obj",
                "type": str,
                "default": "",
                "help": "Legacy replacement object ID (004 or O02@0015@00004); SPIDER tool namespace (045 or kni@1)",
            },
            {
                "name": "--playback_fps",
                "type": float,
                "default": 60.0,
                "help": "Viewer playback rate",
            },
            {
                "name": "--start_frame",
                "type": int,
                "default": 0,
                "help": "First frame to play",
            },
            {
                "name": "--once",
                "action": "store_true",
                "help": "Exit after one playback instead of looping",
            },
        ] + [
            {"name": f"--{entity}_color", "type": _parse_rgb, "default": color,
             "help": f"{entity} RGB as r,g,b with finite channels in [0,1]"}
            for entity, color in DEFAULT_REPLAY_COLORS.items()
        ] + [
            {"name": f"--show_{entity}", "type": _parse_bool, "default": True,
             "help": f"Display {entity}: true or false (default: true)"}
            for entity in ("left_hand", "right_hand", "object", "tool")
        ] + [
            {"name": "--preserve_textures", "type": _parse_bool, "default": False,
             "help": "Keep imported textures and tint with RGB; false uses a neutral white texture"},
        ],
    )
    # gymutil only forwards name/type/default/help, not argparse's choices.
    if args.reference_source not in ("legacy", "spider"):
        raise ValueError("--reference_source must be legacy or spider")
    if args.object_reference not in ("spider_rollout", "demonstration"):
        raise ValueError("--object_reference must be spider_rollout or demonstration")
    return args


def main():
    args = parse_args()
    viewer = None
    try:
        viewer = TacoRetargetViewer(args)
        viewer.run()
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
