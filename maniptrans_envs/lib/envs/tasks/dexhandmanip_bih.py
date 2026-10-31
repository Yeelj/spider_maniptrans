from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from copy import deepcopy
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from lib.utils.visualization import VisualizationConfig
from main.dataset.factory import ManipDataFactory

# from main.dataset.favor_dataset_dexhand import FavorDatasetDexHand
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat
import torch

def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class DexHandManipBiHEnv(VecTask):

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg
        self.visualization = VisualizationConfig.from_mapping(cfg["env"].get("visualization"))
        self._visualization_white_texture = None
        # Keep the existing headless behavior: reference geometry belongs to
        # the viewer, while actor colors also apply to camera sensor images.
        self.show_reference_skeleton = self.visualization.show_reference_skeleton and not headless
        self.reference_source = cfg["env"].get("referenceSource", "legacy")
        if self.reference_source not in ("legacy", "spider"):
            raise ValueError("referenceSource must be legacy or spider")
        if self.reference_source == "spider" and cfg["env"].get("datasetType") != "spider":
            raise ValueError("SPIDER references require task.env.datasetType=spider")

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.dexhand_rh_dof_noise = self.cfg["env"]["dexhand_rDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        self.semantic_contact_eval = None
        self.semantic_contact_detector = None
        self._semantic_contact_snapshot = None
        self.dexhand_rh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")
        self.dexhand_lh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "left")

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand_rh.Kp_rot
            self.Ki_rot = self.dexhand_rh.Ki_rot
            self.Kd_rot = self.dexhand_rh.Kd_rot

            self.Kp_pos = self.dexhand_rh.Kp_pos
            self.Ki_pos = self.dexhand_rh.Ki_pos
            self.Kd_pos = self.dexhand_rh.Kd_pos

        self.cfg["env"]["numActions"] = (
            (1 + 6 + self.dexhand_lh.n_dofs) if use_quat_rot else (6 + self.dexhand_lh.n_dofs)
        ) * (2 if self.cfg["env"]["bimanual_mode"] == "united" else 1)
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.rh_states = {}
        self.lh_states = {}
        self.dexhand_rh_handles = {}  # will be dict mapping names to relevant sim handles
        self.dexhand_lh_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.dataIndices = self.cfg["env"]["dataIndices"]
        self.new_obj = self.cfg["env"].get("new_obj", None)
        if self.reference_source == "spider":
            from lib.utils.spider_reference import normalize_tool
            selected = self.cfg["env"].get("newObj")
            dataset = self.cfg["env"].get("referenceDataset")
            if self.new_obj is not None and normalize_tool(self.new_obj, dataset) != normalize_tool(selected, dataset):
                raise ValueError("new_obj and task.env.newObj disagree; use task.env.newObj in SPIDER mode")
            self.new_obj = selected
        # self.dataIndices = [tuple([int(i) for i in idx.split("@")]) for idx in self.dataIndices]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_rh_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_rh_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_rh_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        TARGET_OBS_DIM = (
            128
            + 5
            + (
                3
                + 3
                + 3
                + 4
                + 4
                + 3
                + 3
                + (self.dexhand_rh.n_bodies - 1) * 9
                + 3
                + 3
                + 3
                + 4
                + 4
                + 3
                + 3
                + self.dexhand_rh.n_bodies
            )
            * self.obs_future_length
        ) * 2
        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        # dexhand_r defaults
        # TODO hack here
        # default_pose = self.cfg["env"].get("dexhand_rDefaultDofPos", None)
        default_pose = torch.ones(self.dexhand_rh.n_dofs, device=self.device) * np.pi / 12
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_rh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)
        self.dexhand_lh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)  # ? TODO check this
        # self.dexhand_rh_default_dof_pos = torch.tensor([-3.5322e-01,  -0.100e-01,  3.2278e-01, -2.51e+00,  1.6036e-01,
        #   2.564e+00, 0.5,  0.10,  0.10], device=self.sim_device)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts_rh = self.demo_data_rh["obj_verts"]
        self.obj_bps_rh = self.bps_layer.encode(obj_verts_rh, feature_type=self.bps_feat_type)[self.bps_feat_type]
        obj_verts_lh = self.demo_data_lh["obj_verts"]
        self.obj_bps_lh = self.bps_layer.encode(obj_verts_lh, feature_type=self.bps_feat_type)[self.bps_feat_type]

        self._init_semantic_contact_eval()

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def _init_semantic_contact_eval(self):
        eval_cfg = self.cfg["env"].get("semanticContactEval", {})
        if self.training or not eval_cfg.get("enabled", False):
            return
        from main.semantic_contact_eval import (
            CollisionSurfaceProxy, SemanticContactAccumulator, SemanticContactEvalConfig,
            full_fragment_end, load_semantic_regions,
            load_contact_required_masks, canonical_contact_metadata_path,
        )
        from main.semantic_contact_geometry import URDFCollisionKinematics
        from pathlib import Path

        config = SemanticContactEvalConfig.from_mapping(eval_cfg)
        # Include every collision link, even those omitted by policy body_names.
        geometry = URDFCollisionKinematics(
            Path(self.dexhand_rh.urdf_path), self.dexhand_rh.dof_names, config.pointsPerLink
        )
        env = self.envs[0]
        actor = self.gym.find_actor_handle(env, "dexhand_r")
        handles = {name: self.gym.find_actor_rigid_body_handle(env, actor, name)
                   for name in geometry.point_link_names}
        missing = [name for name, handle in handles.items() if handle < 0]
        if missing:
            raise ValueError(f"Semantic evaluation collision links missing from simulator: {missing}")
        indices = [handles[geometry.point_link_names[i]] for i in geometry.point_link_ids_array]
        overrides = eval_cfg.get("semanticMeshPaths", {})
        region_cache, regions, region_ids, scales = {}, [], [], []
        for i, env in enumerate(self.envs):
            data_idx = str(self.dataIndices[i % len(self.dataIndices)])
            mesh_path = str(Path(overrides.get(data_idx, self.demo_data_rh["obj_mesh_path"][i])).resolve())
            if mesh_path not in region_cache:
                try:
                    regions.append(load_semantic_regions(mesh_path))
                except (ValueError, OSError) as error:
                    raise ValueError(
                        f"semanticContactEval dataIndices={data_idx}: {error}. Supply a labeled mesh "
                        "of the same tool in meters via semanticMeshPaths; no geometry substitution."
                    ) from error
                region_cache[mesh_path] = len(regions) - 1
            region_ids.append(region_cache[mesh_path])
            tool_actor = self.gym.find_actor_handle(env, "manip_obj_rh")
            scales.append(self.gym.get_actor_scale(env, tool_actor))
        self.semantic_contact_detector = CollisionSurfaceProxy(
            torch.as_tensor(geometry.local_points, device=self.device, dtype=torch.float32),
            torch.tensor(indices, device=self.device), regions,
            torch.tensor(region_ids, device=self.device),
            torch.tensor(scales, device=self.device), config,
        )
        end_frames = full_fragment_end(
            self.demo_data_rh["seq_len"], self.demo_data_lh["seq_len"], self.max_episode_length
        )
        if getattr(self, "reference_source", "legacy") == "spider":
            end_frames = torch.minimum(self.demo_data_rh["seq_len"], self.demo_data_lh["seq_len"]) - 1
        metadata_paths = dict(eval_cfg.get("contactMetadataPaths", {}))
        if config.contactRequiredMaskFile is None:
            for data_idx in self.dataIndices:
                if str(data_idx) not in metadata_paths and self.new_obj is not None:
                    dataset_type = ManipDataFactory.dataset_type(data_idx)
                    dataset_name = {"taco": "TACO", "oakink2": "OakInk-v2"}.get(dataset_type)
                    if dataset_name is not None:
                        path = canonical_contact_metadata_path(
                            config.contactRetargetRoot, dataset_name, str(self.dexhand_rh),
                            str(data_idx), self.new_obj,
                        )
                        if path is not None:
                            metadata_paths[str(data_idx)] = path
        contact_required, self.semantic_contact_mask_sources = load_contact_required_masks(
            [str(self.dataIndices[i % len(self.dataIndices)]) for i in range(self.num_envs)],
            end_frames + 1, self.demo_data_rh.get("tips_distance"), config, metadata_paths,
        )
        self.semantic_contact_eval = SemanticContactAccumulator(end_frames, config, contact_required)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_rh_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4
        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self.dexhand_rh_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_rh_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)
        self.dexhand_lh_pose = deepcopy(self.dexhand_rh_pose)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        if not self.dataIndices:
            raise ValueError("dataIndices must contain at least one task")
        dataset_list = (["spider"] if self.reference_source == "spider" else
                        list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices])))
        if self.new_obj is not None and self.reference_source == "legacy":
            unsupported_datasets = sorted(
                dataset_type for dataset_type in dataset_list if dataset_type not in ("taco", "oakink2")
            )
            if unsupported_datasets:
                raise ValueError(
                    "new_obj only supports TACO and OakInk-V2 bimanual data, but received datasets: "
                    f"{unsupported_datasets}"
                )
            if len(dataset_list) > 1:
                raise ValueError("new_obj requires dataIndices from a single dataset (TACO or OakInk-V2)")

        self.demo_dataset_lh_dict = {}
        self.demo_dataset_rh_dict = {}

        for dataset_type in dataset_list:
            shared_kwargs = {}
            if dataset_type == "spider":
                from hydra.utils import to_absolute_path
                root = self.cfg["env"].get("referenceRoot")
                if not root:
                    raise ValueError("SPIDER mode requires referenceRoot")
                shared_kwargs = dict(
                    reference_root=to_absolute_path(root),
                    reference_dataset=self.cfg["env"].get("referenceDataset"),
                    new_obj=self.new_obj,
                    object_reference=self.cfg["env"].get("objectReference", "demonstration"),
                    allow_failed_reference=self.cfg["env"].get("allowFailedReference", True),
                    asset_root=self.cfg["env"].get("assetRoot"),
                    asset_path_map=self.cfg["env"].get("assetPathMap", {}),
                    control_dt=self.cfg["sim"]["dt"] * self.cfg["env"].get("controlFrequencyInv", 1),
                )
            self.demo_dataset_lh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="left",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_lh,
                embodiment=self.cfg["env"]["dexhand"],
                **shared_kwargs,
            )
            rh_dataset_kwargs = {}
            if dataset_type in ("taco", "oakink2"):
                rh_dataset_kwargs["new_obj"] = self.new_obj
            self.demo_dataset_rh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="right",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_rh,
                embodiment=self.cfg["env"]["dexhand"],
                **shared_kwargs,
                **rh_dataset_kwargs,
            )

        dexhand_rh_asset_file = self.dexhand_rh.urdf_path
        dexhand_lh_asset_file = self.dexhand_lh.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_rh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_rh_asset_file), asset_options)
        dexhand_lh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_lh_asset_file), asset_options)
        if self.reference_source == "spider":
            for side, asset in (("rh", dexhand_rh_asset), ("lh", dexhand_lh_asset)):
                if list(self.gym.get_asset_dof_names(asset)) != list(getattr(self, "dexhand_" + side).dof_names):
                    raise ValueError("PhysX DOF order differs from the training hand: " + side)
        dexhand_rh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_rh_dof_damping = torch.tensor(
            [30] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_damping = torch.tensor(
            [30] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        asset_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }
        self.limit_info["lh"] = {
            "lower": np.asarray(asset_lh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_lh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_rh_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_rh_asset, rigid_shape_rh_props_asset)

        rigid_shape_lh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_lh_asset)
        for element in rigid_shape_lh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_lh_asset, rigid_shape_lh_props_asset)

        self.num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        self.num_dexhand_rh_dofs = self.gym.get_asset_dof_count(dexhand_rh_asset)
        self.num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        self.num_dexhand_lh_dofs = self.gym.get_asset_dof_count(dexhand_lh_asset)

        print(f"Num dexhand_r Bodies: {self.num_dexhand_rh_bodies}")
        print(f"Num dexhand_r DOFs: {self.num_dexhand_rh_dofs}")
        print(f"Num dexhand_l Bodies: {self.num_dexhand_lh_bodies}")
        print(f"Num dexhand_l DOFs: {self.num_dexhand_lh_dofs}")

        dexhand_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        self.dexhand_rh_dof_lower_limits = []
        self.dexhand_rh_dof_upper_limits = []
        self._dexhand_rh_effort_limits = []
        self._dexhand_rh_dof_speed_limits = []
        for i in range(self.num_dexhand_rh_dofs):
            dexhand_rh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_rh_dof_props["stiffness"][i] = dexhand_rh_dof_stiffness[i]
            dexhand_rh_dof_props["damping"][i] = dexhand_rh_dof_damping[i]

            self.dexhand_rh_dof_lower_limits.append(dexhand_rh_dof_props["lower"][i])
            self.dexhand_rh_dof_upper_limits.append(dexhand_rh_dof_props["upper"][i])
            self._dexhand_rh_effort_limits.append(dexhand_rh_dof_props["effort"][i])
            self._dexhand_rh_dof_speed_limits.append(dexhand_rh_dof_props["velocity"][i])

        self.dexhand_rh_dof_lower_limits = torch.tensor(self.dexhand_rh_dof_lower_limits, device=self.sim_device)
        self.dexhand_rh_dof_upper_limits = torch.tensor(self.dexhand_rh_dof_upper_limits, device=self.sim_device)
        self._dexhand_rh_effort_limits = torch.tensor(self._dexhand_rh_effort_limits, device=self.sim_device)
        self._dexhand_rh_dof_speed_limits = torch.tensor(self._dexhand_rh_dof_speed_limits, device=self.sim_device)

        # set dexhand_l dof properties
        dexhand_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.dexhand_lh_dof_lower_limits = []
        self.dexhand_lh_dof_upper_limits = []
        self._dexhand_lh_effort_limits = []
        self._dexhand_lh_dof_speed_limits = []
        for i in range(self.num_dexhand_lh_dofs):
            dexhand_lh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_lh_dof_props["stiffness"][i] = dexhand_lh_dof_stiffness[i]
            dexhand_lh_dof_props["damping"][i] = dexhand_lh_dof_damping[i]

            self.dexhand_lh_dof_lower_limits.append(dexhand_lh_dof_props["lower"][i])
            self.dexhand_lh_dof_upper_limits.append(dexhand_lh_dof_props["upper"][i])
            self._dexhand_lh_effort_limits.append(dexhand_lh_dof_props["effort"][i])
            self._dexhand_lh_dof_speed_limits.append(dexhand_lh_dof_props["velocity"][i])

        self.dexhand_lh_dof_lower_limits = torch.tensor(self.dexhand_lh_dof_lower_limits, device=self.sim_device)
        self.dexhand_lh_dof_upper_limits = torch.tensor(self.dexhand_lh_dof_upper_limits, device=self.sim_device)
        self._dexhand_lh_effort_limits = torch.tensor(self._dexhand_lh_effort_limits, device=self.sim_device)
        self._dexhand_lh_dof_speed_limits = torch.tensor(self._dexhand_lh_dof_speed_limits, device=self.sim_device)

        # compute aggregate size
        num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        num_dexhand_rh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_rh_asset)
        num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        num_dexhand_lh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_lh_asset)

        self.dexhand_rs = []
        self.dexhand_ls = []
        self.envs = []

        assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

        def segment_data(k, data_dict):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            dtype = "spider" if self.reference_source == "spider" else ManipDataFactory.dataset_type(idx)
            return data_dict[dtype][idx]

        self.demo_data_lh = [segment_data(i, self.demo_dataset_lh_dict) for i in tqdm(range(self.num_envs))]
        self.demo_data_lh = self.pack_data(self.demo_data_lh, side="lh")
        self.demo_data_rh = [segment_data(i, self.demo_dataset_rh_dict) for i in tqdm(range(self.num_envs))]
        self.demo_data_rh = self.pack_data(self.demo_data_rh, side="rh")
        if self.reference_source == "spider":
            for key in ("seq_len",):
                if not torch.equal(self.demo_data_rh[key], self.demo_data_lh[key]):
                    raise ValueError("SPIDER hands have different time axes")
            for key in ("manifest_path", "reference_task", "reference_trajectory_sha256", "object_reference", "reference_fps"):
                if self.demo_data_rh[key] != self.demo_data_lh[key]:
                    raise ValueError("SPIDER hands come from different bundles: " + key)
            import warnings
            for side in ("rh", "lh"):
                demo = getattr(self, "demo_data_" + side)
                limits = self.limit_info[side]
                q = demo["opt_dof_pos"]
                lower = torch.as_tensor(limits["lower"], device=q.device)
                upper = torch.as_tensor(limits["upper"], device=q.device)
                violation = torch.maximum(lower - q, q - upper).clamp_min(0).max().item()
                if violation > 0.05:
                    warnings.warn("SPIDER {} maximum DOF limit violation {:.6f} rad; reset clamps execution only, reference is preserved".format(side, violation))

        # Create environments
        self.manip_obj_rh_mass = []
        self.manip_obj_rh_com = []
        self.manip_obj_lh_mass = []
        self.manip_obj_lh_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            rh_current_asset, rh_sum_rigid_body_count, rh_sum_rigid_shape_count, rh_obj_scale, rh_obj_mass = (
                self._create_obj_assets(i, side="rh")
            )
            lh_current_asset, lh_sum_rigid_body_count, lh_sum_rigid_shape_count, lh_obj_scale, lh_obj_mass = (
                self._create_obj_assets(i, side="lh")
            )

            max_agg_bodies = (
                num_dexhand_rh_bodies
                + num_dexhand_lh_bodies
                + 1
                + rh_sum_rigid_body_count
                + lh_sum_rigid_body_count
                + (self.dexhand_rh.n_bodies + self.dexhand_lh.n_bodies if self.show_reference_skeleton else 0)
            )  # 1 for table
            max_agg_shapes = (
                num_dexhand_rh_shapes
                + num_dexhand_lh_shapes
                + 1
                + rh_sum_rigid_shape_count
                + lh_sum_rigid_shape_count
                + (self.dexhand_rh.n_bodies + self.dexhand_lh.n_bodies if self.show_reference_skeleton else 0)
                + (1 if self._record else 0)
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand_r should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            # Create dexhand_r
            dexhand_rh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_rh_asset,
                self.dexhand_rh_pose,
                "dexhand_r",
                i,
                (1 if self.dexhand_rh.self_collision else 0),
            )
            dexhand_lh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_lh_asset,
                self.dexhand_lh_pose,
                "dexhand_l",
                i,
                (1 if self.dexhand_lh.self_collision else 0),
            )
            self._set_actor_color(env_ptr, dexhand_rh_actor, self.visualization.right_hand_color)
            self._set_actor_color(env_ptr, dexhand_lh_actor, self.visualization.left_hand_color)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_rh_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_lh_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_rh_actor, dexhand_rh_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_lh_actor, dexhand_lh_dof_props)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.obj_rh_handle, _ = self._create_obj_actor(
                env_ptr, i, rh_current_asset, side="rh"
            )  # the handle is all the same for all envs
            self.obj_lh_handle, _ = self._create_obj_actor(env_ptr, i, lh_current_asset, side="lh")
            self.gym.set_actor_scale(env_ptr, self.obj_rh_handle, rh_obj_scale)
            self.gym.set_actor_scale(env_ptr, self.obj_lh_handle, lh_obj_scale)
            obj_props_rh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_rh_handle)
            obj_props_lh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_lh_handle)
            obj_props_rh[0].mass = min(0.5, obj_props_rh[0].mass)  # * we only consider the mass less than 500g
            obj_props_lh[0].mass = min(0.5, obj_props_lh[0].mass)  # * we only consider the mass less than 500g

            if rh_obj_mass is not None:
                obj_props_rh[0].mass = rh_obj_mass
            if lh_obj_mass is not None:
                obj_props_lh[0].mass = lh_obj_mass

            # ! Updating the mass and scale might slightly alter the inertia tensor;
            # ! however, because the magnitude of our modifications is minimal, we temporarily neglect this effect.
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_rh_handle, obj_props_rh)
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_lh_handle, obj_props_lh)
            self.manip_obj_rh_mass.append(obj_props_rh[0].mass)
            self.manip_obj_rh_com.append(
                torch.tensor([obj_props_rh[0].com.x, obj_props_rh[0].com.y, obj_props_rh[0].com.z])
            )
            self.manip_obj_lh_mass.append(obj_props_lh[0].mass)
            self.manip_obj_lh_com.append(
                torch.tensor([obj_props_lh[0].com.x, obj_props_lh[0].com.y, obj_props_lh[0].com.z])
            )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexhand_rs.append(dexhand_rh_actor)
            self.dexhand_ls.append(dexhand_lh_actor)

        self.manip_obj_rh_mass = torch.tensor(self.manip_obj_rh_mass, device=self.device)
        self.manip_obj_rh_com = torch.stack(self.manip_obj_rh_com, dim=0).to(self.device)
        self.manip_obj_lh_mass = torch.tensor(self.manip_obj_lh_mass, device=self.device)
        self.manip_obj_lh_com = torch.stack(self.manip_obj_lh_com, dim=0).to(self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_rh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_r")
        dexhand_lh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_l")
        self.dexhand_rh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_rh_handle, k) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_lh_handle, k) for k in self.dexhand_lh.body_names
        }
        # Color APIs require actor-relative body indices, whereas contact
        # tensors above use environment-relative handles.
        self._hand_visual_body_indices = {
            "rh": self.gym.get_actor_rigid_body_dict(env_ptr, dexhand_rh_handle),
            "lh": self.gym.get_actor_rigid_body_dict(env_ptr, dexhand_lh_handle),
        }
        self.dexhand_rh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_lh.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._rh_base_state = self._root_state[:, 0, :]
        self._lh_base_state = self._root_state[:, 1, :]

        # ? >>> for visualization
        if self.show_reference_skeleton:

            self.mano_joint_rh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"rh_mano_joint_{i}"), :]
                for i in range(self.dexhand_rh.n_bodies)
            ]
            self.mano_joint_lh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"lh_mano_joint_{i}"), :]
                for i in range(self.dexhand_lh.n_bodies)
            ]
        # ? <<<

        self._manip_obj_rh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_rh")
        self._manip_obj_rh_root_state = self._root_state[:, self._manip_obj_rh_handle, :]
        self._manip_obj_lh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        self._manip_obj_lh_root_state = self._root_state[:, self._manip_obj_lh_handle, :]
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_rh_handle, "base"
        )
        self._manip_obj_lh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_lh_handle, "base"
        )
        self._manip_obj_rh_cf = self.net_cf[:, self._manip_obj_rh_rigid_body_handle, :]
        self._manip_obj_lh_cf = self.net_cf[:, self._manip_obj_lh_rigid_body_handle, :]

        self.dexhand_rh_root_state = self._root_state[:, dexhand_rh_handle, :]
        self.dexhand_lh_root_state = self._root_state[:, dexhand_lh_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.rh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_r", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_dexhand_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_l", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_rh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_manip_obj_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_lh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        CONTACT_HISTORY_LEN = 3
        self.rh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
        self.lh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()

    def pack_data(self, data, side="rh"):
        spider = data[0].get("reference_source") == "spider"
        if any((d.get("reference_source") == "spider") != spider for d in data):
            raise ValueError("Cannot mix legacy and SPIDER references in one batch")
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = int(packed_data["seq_len"].max().item())
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            stacked = torch.stack(stack_data)
            return stacked if spider else stacked.squeeze()

        for k in data[0].keys():
            if spider and k in ("reference_joint_pos", "reference_joint_velocity", "mano_joints", "mano_joints_velocity"):
                continue
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    if side == "rh":
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_rh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_rh.body_names
                                    if self.dexhand_rh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                    else:
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_lh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_lh.body_names
                                    if self.dexhand_lh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    stacked = torch.stack(stack_data)
                    packed_data[k] = stacked if spider else stacked.squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]
        if spider:
            dexhand = getattr(self, "dexhand_" + side)
            if dexhand.body_names[0] != "palm":
                raise ValueError("Robot references require palm at body index zero")
            # Compatibility field names only: these are distinct robot links,
            # never passed through the many-to-one human to_hand mapping.
            for target, source in (("mano_joints", "reference_joint_pos"),
                                   ("mano_joints_velocity", "reference_joint_velocity")):
                packed_data[target] = fill_data([d[source][:, 1:].reshape(len(d[source]), -1) for d in data])
        return packed_data

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _create_obj_assets(self, i, side="rh"):
        if side == "rh":
            obj_id = self.demo_data_rh["obj_id"][i]
        else:
            obj_id = self.demo_data_lh["obj_id"][i]

        demo = getattr(self, "demo_data_" + side)
        spider = "reference_source" in demo and demo["reference_source"][i] == "spider"
        obj_urdf_path = demo["obj_urdf_path"][i]
        asset_key = ((obj_id, os.path.realpath(obj_urdf_path), demo["object_density"][i],
                      demo["object_friction"][i], demo["object_rolling_friction"][i],
                      demo["object_torsion_friction"][i]) if spider else obj_id)
        if asset_key in self.objs_assets:
            current_asset = self.objs_assets[asset_key]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
            if spider:
                asset_options.density = demo["object_density"][i]
            if side == "rh":
                obj_urdf_path = self.demo_data_rh["obj_urdf_path"][i]
            else:
                obj_urdf_path = self.demo_data_lh["obj_urdf_path"][i]
            current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = demo["object_friction"][i] if spider else 2.0
                element.rolling_friction = demo["object_rolling_friction"][i] if spider else 0.05
                element.torsion_friction = demo["object_torsion_friction"][i] if spider else 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[asset_key] = current_asset

        # * load assigned scale and mass for the object if available
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None
        if spider:
            scale = demo["object_scale"][i]
            mass = demo["object_mass_override"][i]

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def _create_obj_actor(self, env_ptr, i, current_asset, side="rh"):

        if side == "rh":
            obj_transf = self.demo_data_rh["obj_trajectory"][i][0]
        else:
            obj_transf = self.demo_data_lh["obj_trajectory"][i][0]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_quat = rotmat_to_quat(obj_transf[:3, :3])
        pose.r = gymapi.Quat(*[float(v) for v in obj_quat[[1, 2, 3, 0]]])

        # ? object actor filter bit is always 1
        if side == "rh":
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_rh", i, 0)
        else:
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_lh", i, 0)
        self._set_actor_color(
            env_ptr, obj_actor,
            self.visualization.tool_color if side == "rh" else self.visualization.object_color,
        )
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        if side == "rh":
            scene_objs = self.demo_data_rh["scene_objs"][i]
        else:
            scene_objs = self.demo_data_lh["scene_objs"][i]
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True

        for so_id, scene_obj in enumerate(scene_objs):
            scene_obj_type = scene_obj["obj"].type
            scene_obj_size = scene_obj["obj"].size
            scene_obj_pose = scene_obj["pose"]
            if scene_obj_type == "cube":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0],
                    scene_obj_size[1],
                    scene_obj_size[2],
                    scene_asset_options,
                )
                offset = np.eye(4)
                offset[:3, 3] = np.array(scene_obj_size) / 2
                scene_obj_pose = scene_obj_pose @ offset
            elif scene_obj_type == "cylinder":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0] * 2,
                    scene_obj_size[0] * 2,
                    scene_obj_size[1],
                    scene_asset_options,
                )
            else:
                raise NotImplementedError
            scene_obj_pose = self.mujoco2gym_transf @ torch.tensor(
                scene_obj_pose, device=self.sim_device, dtype=torch.float32
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(scene_obj_pose[0, 3], scene_obj_pose[1, 3], scene_obj_pose[2, 3])
            obj_aa = rotmat_to_aa(scene_obj_pose[:3, :3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / obj_aa_angle
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle
            )
            self.gym.create_actor(env_ptr, scene_asset, pose, f"scene_obj_{so_id}", i, 0)
        # add dummy scene object
        MAX_SCENE_OBJS = 0 + (0 if not self.headless else 0)
        for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
            scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
            # ? collision filter bit is always 0b11111111, never collide with anything (except the ground)
            a = self.gym.create_actor(
                env_ptr,
                scene_asset,
                gymapi.Transform(),
                f"scene_obj_{so_id +  len(scene_objs)}",
                self.num_envs + 1,
                0b1,
            )
            c = [
                gymapi.Vec3(1, 1, 0.5),
                gymapi.Vec3(0.5, 1, 1),
                gymapi.Vec3(1, 0, 1),
                gymapi.Vec3(1, 1, 0),
                gymapi.Vec3(0, 1, 1),
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 1, 0),
                gymapi.Vec3(1, 0, 0),
            ][so_id + len(scene_objs)]
            self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        # * just for visualization purposes, add a small sphere at the finger positions
        if self.show_reference_skeleton:
            dexhand_template = self.dexhand_rh if side == "rh" else self.dexhand_lh
            for joint_vis_id, joint_name in enumerate(dexhand_template.body_names):
                joint_name = dexhand_template.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env_ptr,
                    joint_point,
                    gymapi.Transform(),
                    f"{side}_mano_joint_{joint_vis_id}",
                    self.num_envs + 1,
                    0b1,
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        return obj_actor, obj_index

    def _update_states(self):
        self.rh_states.update(
            {
                "q": self._q[:, : self.num_dexhand_rh_dofs],
                "cos_q": torch.cos(self._q[:, : self.num_dexhand_rh_dofs]),
                "sin_q": torch.sin(self._q[:, : self.num_dexhand_rh_dofs]),
                "dq": self._qd[:, : self.num_dexhand_rh_dofs],
                "base_state": self._rh_base_state[:, :],
            }
        )

        self.rh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_rh_handles[k], :][:, :10] for k in self.dexhand_rh.body_names],
            dim=1,
        )
        self.rh_states.update(
            {
                "manip_obj_pos": self._manip_obj_rh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_rh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_rh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_rh_root_state[:, 10:],
            }
        )

        self.lh_states.update(
            {
                "q": self._q[:, self.num_dexhand_rh_dofs :],
                "cos_q": torch.cos(self._q[:, self.num_dexhand_rh_dofs :]),
                "sin_q": torch.sin(self._q[:, self.num_dexhand_rh_dofs :]),
                "dq": self._qd[:, self.num_dexhand_rh_dofs :],
                "base_state": self._lh_base_state[:, :],
            }
        )
        self.lh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_lh_handles[k], :][:, :10] for k in self.dexhand_lh.body_names],
            dim=1,
        )
        self.lh_states.update(
            {
                "manip_obj_pos": self._manip_obj_lh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_lh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_lh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_lh_root_state[:, 10:],
            }
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        (
            lh_rew_buf,
            lh_reset_buf,
            lh_success_buf,
            lh_failure_buf,
            lh_reward_dict,
            lh_error_buf,
            lh_diff_obj_pos,
            lh_diff_obj_rot,
            lh_diff_joints,
            lh_diff_ft,
        ) = self.compute_reward_side(actions, side="lh")
        (
            rh_rew_buf,
            rh_reset_buf,
            rh_success_buf,
            rh_failure_buf,
            rh_reward_dict,
            rh_error_buf,
            rh_diff_obj_pos,
            rh_diff_obj_rot,
            rh_diff_joints,
            rh_diff_ft,
        ) = self.compute_reward_side(actions, side="rh")
        self.rew_buf = rh_rew_buf + lh_rew_buf
        self.diff_obj_pos = (rh_diff_obj_pos + lh_diff_obj_pos) / 2
        self.diff_obj_rot = (rh_diff_obj_rot + lh_diff_obj_rot) / 2
        self.diff_joints = (rh_diff_joints + lh_diff_joints) / 2
        self.diff_ft = (rh_diff_ft + lh_diff_ft) / 2
        self.reset_buf = rh_reset_buf | lh_reset_buf
        self.success_buf = rh_success_buf & lh_success_buf
        self.failure_buf = rh_failure_buf | lh_failure_buf
        self.error_buf = rh_error_buf | lh_error_buf
        self.reward_dict = {
            **{"rh_" + k: v for k, v in rh_reward_dict.items()},
            **{"lh_" + k: v for k, v in lh_reward_dict.items()},
        }

    def compute_reward_side(self, actions, side="rh"):
        side_demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        target_state = {}
        max_length = torch.clip(side_demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = torch.minimum(self.progress_buf, side_demo_data["seq_len"] - 1)
        cur_wrist_pos = side_demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_pos"] = cur_wrist_pos
        cur_wrist_rot = side_demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = side_demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = side_demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tips_distance"] = side_demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]

        cur_joints_pos = side_demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = side_demo_data["mano_joints_velocity"][
            torch.arange(self.num_envs), cur_idx
        ].reshape(self.num_envs, -1, 3)

        cur_obj_transf = side_demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
        target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]

        target_state["manip_obj_vel"] = side_demo_data["obj_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_ang_vel"] = side_demo_data["obj_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tip_force"] = torch.stack(
            [
                self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                for k in (self.dexhand_rh.contact_body_names if side == "rh" else self.dexhand_lh.contact_body_names)
            ],
            axis=1,
        )
        setattr(
            self,
            f"{side}_tips_contact_history",
            torch.concat(
                [
                    getattr(self, f"{side}_tips_contact_history")[:, 1:],
                    (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
                ],
                dim=1,
            ),
        )
        target_state["tip_contact_state"] = getattr(self, f"{side}_tips_contact_history")

        side_states = getattr(self, f"{side}_states")
        if side == "rh":
            power = torch.abs(torch.multiply(self.dof_force[:, : self.dexhand_rh.n_dofs], side_states["dq"])).sum(
                dim=-1
            )
        else:
            power = torch.abs(torch.multiply(self.dof_force[:, self.dexhand_rh.n_dofs :], side_states["dq"])).sum(
                dim=-1
            )
        target_state["power"] = power

        base_handle = getattr(self, f"dexhand_{side}_handles")[
            self.dexhand_rh.to_dex("wrist")[0] if side == "rh" else self.dexhand_lh.to_dex("wrist")[0]
        ]

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, base_handle, :] * side_states["base_state"][:, 7:10],
                dim=-1,
            )
        )  # ? linear force * linear velocity
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, base_handle, :] * side_states["base_state"][:, 10:],
                dim=-1,
            )
        )  # ? torque * angular velocity
        target_state["wrist_power"] = wrist_power

        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError
        else:
            scale_factor = 1.0

        assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            margin = 0 if self.reference_source == "spider" else 3
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + margin + 1)

        (
            rew_buf,
            reset_buf,
            success_buf,
            failure_buf,
            reward_dict,
            error_buf,
            diff_obj_pos,
            diff_obj_rot,
            diff_joints,
            diff_ft,
        ) = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            side_states,
            target_state,
            max_length,
            scale_factor,
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
            0 if self.reference_source == "spider" else 3,
        )
        self.total_rew_buf += rew_buf
        return (
            rew_buf,
            reset_buf,
            success_buf,
            failure_buf,
            reward_dict,
            error_buf,
            diff_obj_pos,
            diff_obj_rot,
            diff_joints,
            diff_ft,
        )

    def compute_observations(self):
        self._refresh()
        obs_rh = self.compute_observations_side("rh")
        obs_lh = self.compute_observations_side("lh")
        for k in obs_rh.keys():
            self.obs_dict[k] = torch.cat([obs_rh[k], obs_lh[k]], dim=-1)

    def compute_observations_side(self, side="rh"):
        # obs_keys: q, cos_q, sin_q, base_state
        side_states = getattr(self, f"{side}_states")
        side_demo_data = getattr(self, f"demo_data_{side}")

        obs_dict = {}

        obs_values = []
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(side_states[ob][:, :3]), side_states[ob][:, 3:]], dim=-1)
                )  # ! ignore base position
            else:
                obs_values.append(side_states[ob])
        obs_dict["proprioception"] = torch.cat(obs_values, dim=-1)
        # privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, manip_obj_vel, manip_obj_ang_vel
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_values.append(side_states[ob] - side_states["base_state"][:, :3])
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(side_states["manip_obj_quat"][:, [1, 2, 3, 0]])
                        @ getattr(self, f"manip_obj_{side}_com").unsqueeze(-1)
                    ).squeeze(-1) + side_states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - side_states["base_state"][:, :3])
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((getattr(self, f"manip_obj_{side}_mass") * -1 * prop.gravity.z).unsqueeze(-1))
                elif ob == "tip_force":
                    tip_force = torch.stack(
                        [
                            self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                            for k in (
                                self.dexhand_rh.contact_body_names
                                if side == "rh"
                                else self.dexhand_lh.contact_body_names
                            )
                        ],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                else:
                    pri_obs_values.append(side_states[ob])
            obs_dict["privileged"] = torch.cat(pri_obs_values, dim=-1)

        next_target_state = {}

        cur_idx = self.progress_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(side_demo_data["seq_len"]), side_demo_data["seq_len"] - 1)

        cur_idx = torch.stack(
            [cur_idx + t for t in range(self.obs_future_length)], dim=-1
        )  # [B, K], K = obs_future_length
        cur_idx = torch.minimum(cur_idx, (side_demo_data["seq_len"] - 1)[:, None])
        nE, nT = side_demo_data["wrist_pos"].shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        target_wrist_pos = indicing(side_demo_data["wrist_pos"], cur_idx)  # [B, K, 3]
        cur_wrist_pos = side_states["base_state"][:, :3]  # [B, 3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(side_demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = side_states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(side_demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot = side_states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(side_demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = side_states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(side_demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = side_states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(side_demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = side_states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obj_transf = indicing(side_demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - side_states["manip_obj_pos"][:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(side_demo_data["obj_velocity"], cur_idx)
        cur_obj_vel = side_states["manip_obj_vel"]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)

        next_target_state["manip_obj_quat"] = rotmat_to_quat(target_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            side_states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(nE, -1)
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(nE, -1)

        target_obj_ang_vel = indicing(side_demo_data["obj_angular_velocity"], cur_idx)
        cur_obj_ang_vel = side_states["manip_obj_ang_vel"]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)

        next_target_state["obj_to_joints"] = torch.norm(
            side_states["manip_obj_pos"][:, None] - side_states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)

        next_target_state["gt_tips_distance"] = indicing(side_demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        next_target_state["bps"] = getattr(self, f"obj_bps_{side}")
        obs_dict["target"] = torch.cat(
            [
                next_target_state[ob]
                for ob in [  # ! must be in the same order as the following
                    "delta_wrist_pos",
                    "wrist_vel",
                    "delta_wrist_vel",
                    "wrist_quat",
                    "delta_wrist_quat",
                    "wrist_ang_vel",
                    "delta_wrist_ang_vel",
                    "delta_joints_pos",
                    "joints_vel",
                    "delta_joints_vel",
                    "delta_manip_obj_pos",
                    "manip_obj_vel",
                    "delta_manip_obj_vel",
                    "manip_obj_quat",
                    "delta_manip_obj_quat",
                    "manip_obj_ang_vel",
                    "delta_manip_obj_ang_vel",
                    "obj_to_joints",
                    "gt_tips_distance",
                    "bps",
                ]
            ],
            dim=-1,
        )

        if not self.training:
            manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")
            dexhand_handles = getattr(self, f"dexhand_{side}_handles")
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_manip_obj_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "state_manip_obj_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "joint_state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_rh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "joint_state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_lh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "q_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "q_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "dq_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "dq_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "tip_force_rh" and side == "rh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_rh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "tip_force_lh" and side == "lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_lh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:
                    pass
        return obs_dict

    def _reset_default(self, env_ids):
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data_rh["seq_len"][env_ids]
                    * 0.98
                    * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())
            else:
                seq_idx = torch.zeros_like(self.demo_data_rh["seq_len"][env_ids].long())

        self._reset_default_side(env_ids, seq_idx, side="lh")
        self._reset_default_side(env_ids, seq_idx, side="rh")

        dexhand_multi_env_ids_int32 = torch.concat(
            [
                self._global_dexhand_rh_indices[env_ids].flatten(),
                self._global_dexhand_lh_indices[env_ids].flatten(),
            ]
        )
        manip_obj_multi_env_ids_int32 = torch.concat(
            [self._global_manip_obj_rh_indices[env_ids].flatten(), self._global_manip_obj_lh_indices[env_ids].flatten()]
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0

        if self.use_pid_control:
            self.rh_prev_pos_error[env_ids] = 0
            self.rh_prev_rot_error[env_ids] = 0
            self.rh_pos_error_integral[env_ids] = 0
            self.rh_rot_error_integral[env_ids] = 0
            self.lh_prev_pos_error[env_ids] = 0
            self.lh_prev_rot_error[env_ids] = 0
            self.lh_pos_error_integral[env_ids] = 0
            self.lh_rot_error_integral[env_ids] = 0

        self.lh_tips_contact_history[env_ids] = torch.ones_like(self.lh_tips_contact_history[env_ids]).bool()
        self.rh_tips_contact_history[env_ids] = torch.ones_like(self.rh_tips_contact_history[env_ids]).bool()

    def _reset_default_side(self, env_ids, seq_idx, side="rh"):

        side_demo_data = getattr(self, f"demo_data_{side}")

        dof_pos = side_demo_data["opt_dof_pos"][env_ids, seq_idx]
        dof_pos = torch_jit_utils.tensor_clamp(
            dof_pos,
            getattr(self, f"dexhand_{side}_dof_lower_limits").unsqueeze(0),
            getattr(self, f"dexhand_{side}_dof_upper_limits").unsqueeze(0),
        )
        dof_vel = side_demo_data["opt_dof_velocity"][env_ids, seq_idx]
        dof_vel = torch_jit_utils.tensor_clamp(
            dof_vel,
            -1 * getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
            getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
        )

        opt_wrist_pos = side_demo_data["opt_wrist_pos"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_quat(side_demo_data["opt_wrist_rot"][env_ids, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = side_demo_data["opt_wrist_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = side_demo_data["opt_wrist_angular_velocity"][env_ids, seq_idx]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        getattr(self, f"_{side}_base_state")[env_ids, :] = opt_hand_pose_vel

        if side == "rh":
            self._q[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
            self._qd[env_ids, : self.num_dexhand_rh_dofs] = dof_vel
            self._pos_control[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
        else:
            self._q[env_ids, self.num_dexhand_rh_dofs :] = dof_pos
            self._qd[env_ids, self.num_dexhand_rh_dofs :] = dof_vel
            self._pos_control[env_ids, self.num_dexhand_rh_dofs :] = dof_pos

        # reset manip obj
        obj_pos_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        obj_rot_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = side_demo_data["obj_velocity"][env_ids, seq_idx]
        obj_ang_vel = side_demo_data["obj_angular_velocity"][env_ids, seq_idx]

        manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")

        manip_obj_root_state[env_ids, :3] = obj_pos_init
        manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        manip_obj_root_state[env_ids, 7:10] = obj_vel
        manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        self._reset_default(env_ids)
        if self.semantic_contact_eval is not None:
            self.semantic_contact_eval.reset(
                env_ids, self.progress_buf[env_ids], eligible=not self.random_state_init
            )

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        if self.semantic_contact_eval is not None:
            info["semantic_contact_eval"] = self._semantic_contact_snapshot
        info["diff_metrics"] = {
            "diff_obj_pos": self.diff_obj_pos,
            "diff_obj_rot": self.diff_obj_rot,
            "diff_joints": self.diff_joints,
            "diff_ft": self.diff_ft,
        }
        return obs, rew, done, info

    def _set_actor_color(self, env_ptr, actor, color):
        if color is None or self.graphics_device_id < 0:
            return
        if not self.visualization.preserve_textures and self._visualization_white_texture is None:
            # A neutral texture prevents imported mesh textures from tinting
            # custom colors. INVALID_HANDLE is not a supported removal API in
            # this SDK and can produce incorrect camera colors.
            self._visualization_white_texture = self.gym.create_texture_from_buffer(
                self.sim, 1, 1, np.full((1, 4), 255, dtype=np.uint8)
            )
            if self._visualization_white_texture == gymapi.INVALID_HANDLE:
                raise RuntimeError("Failed to create the neutral visualization texture")
        for body_index in range(self.gym.get_actor_rigid_body_count(env_ptr, actor)):
            if not self.visualization.preserve_textures:
                self.gym.set_rigid_body_texture(
                    env_ptr, actor, body_index, gymapi.MESH_VISUAL, self._visualization_white_texture
                )
            self.gym.set_rigid_body_color(env_ptr, actor, body_index, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

    def _update_visualization(self):
        if self.headless:
            return

        # Contact highlighting is independent of the reference skeleton.
        if self.visualization.show_contact_highlight:
            for side in ("lh", "rh"):
                hand = getattr(self, f"dexhand_{side}")
                handles = getattr(self, f"dexhand_{side}_handles")
                for env_id, env_ptr in enumerate(self.envs):
                    for name in hand.body_names:
                        has_force = torch.norm(self.net_cf[env_id, handles[name]], dim=-1) != 0
                        self.set_force_vis(env_ptr, name, has_force, side)

        if not self.show_reference_skeleton:
            return

        self.gym.clear_lines(self.viewer)
        color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        for side in ("lh", "rh"):
            hand = getattr(self, f"dexhand_{side}")
            demo = getattr(self, f"demo_data_{side}")
            cur_idx = torch.minimum(self.progress_buf, demo["seq_len"] - 1)
            env_ids = torch.arange(self.num_envs, device=cur_idx.device)
            wrist_pos = demo["wrist_pos"][env_ids, cur_idx]
            joint_pos = demo["mano_joints"][env_ids, cur_idx].reshape(self.num_envs, -1, 3)
            joint_pos = torch.cat([wrist_pos[:, None], joint_pos], dim=1)
            for k, point in enumerate(getattr(self, f"mano_joint_{side}_points")):
                point[:, :3] = joint_pos[:, k]
            for env_id, env_ptr in enumerate(self.envs):
                hand_joints = joint_pos[env_id].cpu().numpy()
                assert hand_joints.shape == (hand.n_bodies, 3)
                for start, end in hand.bone_links:
                    line = np.array([hand_joints[start], hand_joints[end]])
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, color)

    def pre_physics_step(self, actions):
        self._update_visualization()

        root_control_dim = 9 if self.use_pid_control else 6
        res_split_idx = (
            actions.shape[1] // 2
            if not self.use_pid_control
            else ((actions.shape[1] - 2 * (root_control_dim - 6)) // 2) + 2 * (root_control_dim - 6)
        )

        base_action = actions[:, :res_split_idx]  # ? in the range of [-1, 1]
        residual_action = actions[:, res_split_idx:] * 2  # ? the delta action is theoritically in the range of [-2, 2]

        rh_dof_pos = (
            1.0 * base_action[:, root_control_dim : root_control_dim + self.num_dexhand_rh_dofs]
            + residual_action[:, 6 : 6 + self.num_dexhand_rh_dofs]
        )
        rh_dof_pos = torch.clamp(rh_dof_pos, -1, 1)

        lh_dof_pos = (
            1.0 * base_action[:, root_control_dim + root_control_dim + self.num_dexhand_rh_dofs :]
            + residual_action[:, 6 + 6 + self.num_dexhand_rh_dofs :]
        )
        lh_dof_pos = torch.clamp(lh_dof_pos, -1, 1)

        curr_act_moving_average = self.act_moving_average

        self.rh_curr_targets = torch_jit_utils.scale(
            rh_dof_pos,  # ! actions must in [-1, 1]
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.rh_curr_targets = (
            curr_act_moving_average * self.rh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, : self.num_dexhand_rh_dofs]
        )
        self.rh_curr_targets = torch_jit_utils.tensor_clamp(
            self.rh_curr_targets,
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.prev_targets[:, : self.num_dexhand_rh_dofs] = self.rh_curr_targets[:]

        self.lh_curr_targets = torch_jit_utils.scale(
            lh_dof_pos,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.lh_curr_targets = (
            curr_act_moving_average * self.lh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, self.num_dexhand_rh_dofs :]
        )
        self.lh_curr_targets = torch_jit_utils.tensor_clamp(
            self.lh_curr_targets,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.prev_targets[:, self.num_dexhand_rh_dofs :] = self.lh_curr_targets[:]

        if self.use_pid_control:
            rh_position_error = base_action[:, 0:3]
            self.rh_pos_error_integral += rh_position_error * self.dt
            self.rh_pos_error_integral = torch.clamp(self.rh_pos_error_integral, -1, 1)
            rh_pos_derivative = (rh_position_error - self.rh_prev_pos_error) / self.dt
            rh_force = (
                self.Kp_pos * rh_position_error
                + self.Ki_pos * self.rh_pos_error_integral
                + self.Kd_pos * rh_pos_derivative
            )
            self.rh_prev_pos_error = rh_position_error

            rh_force = rh_force + residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_position_error = base_action[
                :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
            ]
            self.lh_pos_error_integral += lh_position_error * self.dt
            self.lh_pos_error_integral = torch.clamp(self.lh_pos_error_integral, -1, 1)
            lh_pos_derivative = (lh_position_error - self.lh_prev_pos_error) / self.dt
            lh_force = (
                self.Kp_pos * lh_position_error
                + self.Ki_pos * self.lh_pos_error_integral
                + self.Kd_pos * lh_pos_derivative
            )
            self.lh_prev_pos_error = lh_position_error

            lh_force = (
                lh_force
                + residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

            rh_rotation_error = base_action[:, 3:root_control_dim]
            rh_rotation_error = rot6d_to_aa(rh_rotation_error)
            self.rh_rot_error_integral += rh_rotation_error * self.dt
            self.rh_rot_error_integral = torch.clamp(self.rh_rot_error_integral, -1, 1)
            rh_rot_derivative = (rh_rotation_error - self.rh_prev_rot_error) / self.dt
            rh_torque = (
                self.Kp_rot * rh_rotation_error
                + self.Ki_rot * self.rh_rot_error_integral
                + self.Kd_rot * rh_rot_derivative
            )
            self.rh_prev_rot_error = rh_rotation_error

            rh_torque = rh_torque + residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_rotation_error = base_action[
                :,
                root_control_dim
                + self.num_dexhand_rh_dofs
                + 3 : root_control_dim
                + self.num_dexhand_rh_dofs
                + root_control_dim,
            ]
            lh_rotation_error = rot6d_to_aa(lh_rotation_error)
            self.lh_rot_error_integral += lh_rotation_error * self.dt
            self.lh_rot_error_integral = torch.clamp(self.lh_rot_error_integral, -1, 1)
            lh_rot_derivative = (lh_rotation_error - self.lh_prev_rot_error) / self.dt
            lh_torque = (
                self.Kp_rot * lh_rotation_error
                + self.Ki_rot * self.lh_rot_error_integral
                + self.Kd_rot * lh_rot_derivative
            )
            self.lh_prev_rot_error = lh_rotation_error

            lh_torque = (
                lh_torque
                + residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
        else:
            rh_force = 1.0 * (base_action[:, 0:3] * self.dt * self.translation_scale * 500) + (
                residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            )
            rh_torque = 1.0 * (base_action[:, 3:6] * self.dt * self.orientation_scale * 200) + (
                residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            )
            lh_force = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
                ]
                * self.dt
                * self.translation_scale
                * 500
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            lh_torque = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs + 3 : root_control_dim + self.num_dexhand_rh_dofs + 6
                ]
                * self.dt
                * self.orientation_scale
                * 200
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )

            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.semantic_contact_eval is not None:
            red_events, black_events = self.semantic_contact_detector.detect(
                self._rigid_body_state, self.net_cf, self._manip_obj_rh_root_state,
                self.semantic_contact_eval.active(self.progress_buf),
            )
            self.semantic_contact_eval.record(self.progress_buf, red_events, black_events)
            # Clone terminal results before progress increments or selective resets.
            self._semantic_contact_snapshot = self.semantic_contact_eval.finish(
                self.reset_buf, self.success_buf
            )

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_force_vis(self, env_ptr, part_k, has_force, side):
        self.gym.set_rigid_body_color(
            env_ptr,
            self.gym.find_actor_handle(env_ptr, "dexhand_l" if side == "lh" else "dexhand_r"),
            self._hand_visual_body_indices[side][part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(*(self.visualization.hand_color(side) or (1.0, 1.0, 1.0)))
            ),
        )


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
    terminal_margin: int = 3,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[str, Tensor],
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float, Dict[str, List[int]], int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    # object pose reward
    current_obj_pos = states["manip_obj_pos"]
    current_obj_quat = states["manip_obj_quat"]

    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    finger_tip_force = target_states["tip_force"]
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )  # sanity check

    diff_joints = torch.stack([diff_level_1_pos_dist, diff_level_2_pos_dist]).mean(dim=0) * 100.0
    diff_ft = torch.stack(
        [
            diff_thumb_tip_pos_dist,
            diff_index_tip_pos_dist,
            diff_middle_tip_pos_dist,
            diff_pinky_tip_pos_dist,
            diff_ring_tip_pos_dist,
        ]
    ).mean(dim=0) * 100.0
    diff_obj_pos = diff_obj_pos_dist * 100.0
    diff_obj_rot = diff_obj_rot_angle.abs() / np.pi * 180

    failed_execute = (
        (
            (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor**3)  # TODO
            | (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
            | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
            | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
            | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor**3)  # TODO
            | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
        )
        & (running_progress_buf >= 8)
    ) | error_buf
    reward_execute = (
        0.1 * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.3 * reward_level_2_pos
        + 5.0 * reward_obj_pos
        + 1.0 * reward_obj_rot
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        + 0.1 * reward_obj_ang_vel
        + 1.0 * reward_finger_tip_force
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 + terminal_margin >= max_length
    ) & ~failed_execute  # legacy margin stays 3; SPIDER includes the real last frame
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_finger_tip_force": reward_finger_tip_force,
    }

    return (
        reward_execute,
        reset_buf,
        succeeded,
        failed_execute,
        reward_dict,
        error_buf,
        diff_obj_pos,
        diff_obj_rot,
        diff_joints,
        diff_ft,
    )
