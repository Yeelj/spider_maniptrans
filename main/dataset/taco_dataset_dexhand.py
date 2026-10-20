import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from manotorch.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation, Slerp

from main.dataset.transform import aa_to_rotmat

from .base import ManipData
from .decorators import register_manipdata
from .taco_dataset_utils import build_taco_sequence_index, taco_asset_paths


TACO_SOURCE_FPS = 30
MANIPTRANS_FPS = 60
TACO_MESH_SCALE = 0.01  # TACO's *_cm.obj files use centimeters.


def _load_mesh(mesh_path: str, scale: float = 1.0) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False, skip_materials=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {mesh_path}, got {type(mesh).__name__}")
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    return mesh


def _load_meter_mesh(mesh_path: str) -> trimesh.Trimesh:
    return _load_mesh(mesh_path, scale=TACO_MESH_SCALE)


def _normalize_new_object_id(new_obj):
    if new_obj is None or new_obj == "":
        return None
    if isinstance(new_obj, (bool, np.bool_)):
        raise ValueError(f"new_obj must be a three-digit TACO object ID, got {new_obj!r}")
    if isinstance(new_obj, (int, np.integer)):
        object_number = int(new_obj)
    elif isinstance(new_obj, str) and new_obj.isdigit():
        object_number = int(new_obj)
    else:
        raise ValueError(f"new_obj must be a three-digit TACO object ID, got {new_obj!r}")
    if not 0 <= object_number <= 999:
        raise ValueError(f"new_obj must be between 000 and 999, got {new_obj!r}")
    return f"{object_number:03d}"


def _linear_resample(data: torch.Tensor, source_fps: int, target_fps: int) -> torch.Tensor:
    if source_fps == target_fps:
        return data
    if target_fps % source_fps != 0:
        raise ValueError(f"Only integer upsampling is supported, got {source_fps} -> {target_fps} FPS")
    if data.shape[0] < 2:
        raise ValueError("A TACO sequence must contain at least two frames")
    target_length = (data.shape[0] - 1) * (target_fps // source_fps) + 1
    flattened = data.reshape(data.shape[0], -1).transpose(0, 1).unsqueeze(0)
    resampled = F.interpolate(flattened, size=target_length, mode="linear", align_corners=True)
    return resampled.squeeze(0).transpose(0, 1).reshape(target_length, *data.shape[1:])


def _rotation_resample(rotation_matrices: torch.Tensor, source_fps: int, target_fps: int) -> torch.Tensor:
    if source_fps == target_fps:
        return rotation_matrices
    if target_fps % source_fps != 0:
        raise ValueError(f"Only integer upsampling is supported, got {source_fps} -> {target_fps} FPS")
    source_length = rotation_matrices.shape[0]
    if source_length < 2:
        raise ValueError("A TACO sequence must contain at least two frames")
    target_length = (source_length - 1) * (target_fps // source_fps) + 1
    source_time = np.arange(source_length, dtype=np.float64) / source_fps
    target_time = np.arange(target_length, dtype=np.float64) / target_fps
    matrices = rotation_matrices.detach().cpu().numpy()
    interpolated = Slerp(source_time, Rotation.from_matrix(matrices))(target_time).as_matrix()
    return torch.as_tensor(interpolated, dtype=rotation_matrices.dtype, device=rotation_matrices.device)


def _pose_resample(poses: torch.Tensor, source_fps: int, target_fps: int) -> torch.Tensor:
    translations = _linear_resample(poses[:, :3, 3], source_fps, target_fps)
    rotations = _rotation_resample(poses[:, :3, :3], source_fps, target_fps)
    result = torch.eye(4, dtype=poses.dtype, device=poses.device).repeat(translations.shape[0], 1, 1)
    result[:, :3, :3] = rotations
    result[:, :3, 3] = translations
    return result


def _validate_rigid_trajectory(name: str, trajectory: np.ndarray) -> None:
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [T, 4, 4], got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} contains NaN or Inf")
    rotations = trajectory[:, :3, :3]
    identity = np.eye(3, dtype=trajectory.dtype)[None]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), identity, atol=1e-4):
        raise ValueError(f"{name} contains non-orthonormal rotations")
    if not np.allclose(trajectory[:, 3], np.array([0, 0, 0, 1], dtype=trajectory.dtype), atol=1e-6):
        raise ValueError(f"{name} contains invalid homogeneous transforms")


class TacoDatasetDexHand(ManipData):
    side = None

    def __init__(
        self,
        *,
        data_dir: str = "data/TACO",
        split: str = "all",
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        mano_assets_root: str = "data/mano_v1_2",
        workspace_center=(-0.1, 0.0),
        floor_clearance: float = 0.003,
        max_hand_object_distance: float = 0.45,
        new_obj=None,
        **kwargs,
    ):
        if self.side not in ("left", "right"):
            raise ValueError("TacoDatasetDexHand must be specialized with side='left' or side='right'")
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=1,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            output_fps=MANIPTRANS_FPS,
            **kwargs,
        )
        self.sequence_index = build_taco_sequence_index(data_dir)
        self.data_pathes = [self.sequence_index[key].hand_pose_dir for key in sorted(self.sequence_index)]
        self._base_mujoco2gym_transf = mujoco2gym_transf.clone()
        self.workspace_center = np.asarray(workspace_center, dtype=np.float32)
        if self.workspace_center.shape != (2,):
            raise ValueError(f"workspace_center must contain two values, got {workspace_center}")
        self.floor_clearance = float(floor_clearance)
        self.max_hand_object_distance = float(max_hand_object_distance)
        self.new_obj = _normalize_new_object_id(new_obj)
        if self.side == "left" and self.new_obj is not None:
            raise ValueError("new_obj can only replace the right-hand TACO tool mesh")
        if self.new_obj is not None:
            override_paths = taco_asset_paths(self.data_dir, self.new_obj)
            missing_override_assets = [path for path in override_paths if not os.path.isfile(path)]
            if missing_override_assets:
                raise FileNotFoundError(
                    f"Missing processed TACO new_obj={self.new_obj} assets: {missing_override_assets}. Run "
                    f"`python scripts/preprocess_taco_assets.py --data-dir {self.data_dir} "
                    f"--object-id {self.new_obj}` first."
                )

        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side=self.side,
            center_idx=0,
            mano_assets_root=mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)
        self.manolayer.eval()

        # ManipTrans's retargeter always applies this source-to-Gym rotation.
        # TACO is already z-up, so pre-applying its inverse keeps TACO's z axis
        # aligned with gravity both during standalone retargeting and RL loading.
        source_to_gym = aa_to_rotmat(np.array([0, 0, -np.pi / 2], dtype=np.float32)) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0], dtype=np.float32)
        )
        correction = torch.eye(4, dtype=torch.float32, device=device)
        correction[:3, :3] = torch.as_tensor(source_to_gym.T, dtype=torch.float32, device=device)
        self._source_frame_correction = correction

    def __len__(self):
        return len(self.sequence_index)

    def _load_hand(self, record):
        pose_path = os.path.join(record.hand_pose_dir, f"{self.side}_hand.pkl")
        shape_path = os.path.join(record.hand_pose_dir, f"{self.side}_hand_shape.pkl")
        for path in (pose_path, shape_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing TACO hand annotation: {path}")

        with open(pose_path, "rb") as stream:
            hand_data = pickle.load(stream)
        with open(shape_path, "rb") as stream:
            shape_data = pickle.load(stream)
        frame_keys = sorted(hand_data)
        poses = torch.stack([hand_data[key]["hand_pose"].detach().float() for key in frame_keys]).to(self.device)
        translations = torch.stack([hand_data[key]["hand_trans"].detach().float() for key in frame_keys]).to(
            self.device
        )
        betas = shape_data["hand_shape"].detach().float().reshape(1, 10).to(self.device)
        betas = betas.repeat(poses.shape[0], 1)
        with torch.no_grad():
            mano_output = self.manolayer(poses, betas)
        # center_idx=0 returns a wrist-centered MANO hand; TACO hand_trans is
        # the metric world-space wrist translation.
        joints = mano_output.joints + translations[:, None]
        return pose_path, poses, translations, joints

    @staticmethod
    def _mano_joint_dict(joints: torch.Tensor):
        # manotorch SNAP order: wrist, thumb(4), index(4), middle(4),
        # ring(4), pinky(4).
        return {
            "thumb_proximal": joints[:, 1],
            "thumb_intermediate": joints[:, 2],
            "thumb_distal": joints[:, 3],
            "thumb_tip": joints[:, 4],
            "index_proximal": joints[:, 5],
            "index_intermediate": joints[:, 6],
            "index_distal": joints[:, 7],
            "index_tip": joints[:, 8],
            "middle_proximal": joints[:, 9],
            "middle_intermediate": joints[:, 10],
            "middle_distal": joints[:, 11],
            "middle_tip": joints[:, 12],
            "ring_proximal": joints[:, 13],
            "ring_intermediate": joints[:, 14],
            "ring_distal": joints[:, 15],
            "ring_tip": joints[:, 16],
            "pinky_proximal": joints[:, 17],
            "pinky_intermediate": joints[:, 18],
            "pinky_distal": joints[:, 19],
            "pinky_tip": joints[:, 20],
        }

    def _normalization_transform(self, tool_trajectory, target_trajectory, tool_mesh, target_mesh):
        first_frame_count = min(5, len(tool_trajectory))
        object_xy = np.concatenate(
            [tool_trajectory[:first_frame_count, :2, 3], target_trajectory[:first_frame_count, :2, 3]], axis=0
        )
        xy_anchor = np.median(object_xy, axis=0)

        min_z = np.inf
        for trajectory, mesh in ((tool_trajectory, tool_mesh), (target_trajectory, target_mesh)):
            for pose in trajectory[:first_frame_count]:
                vertices = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
                min_z = min(min_z, float(vertices[:, 2].min()))

        normalization = torch.eye(4, dtype=torch.float32, device=self.device)
        normalization[:2, 3] = torch.as_tensor(self.workspace_center - xy_anchor, device=self.device)
        normalization[2, 3] = self.floor_clearance - min_z
        return normalization

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        if not isinstance(index, str) or index not in self.sequence_index:
            example = "bb_bowl_0919036"
            raise KeyError(f"Unknown TACO sequence ID {index!r}; expected an ID such as {example}")
        record = self.sequence_index[index]

        tool_trajectory_np = np.load(record.tool_pose_path).astype(np.float32)
        target_trajectory_np = np.load(record.target_pose_path).astype(np.float32)
        _validate_rigid_trajectory("tool trajectory", tool_trajectory_np)
        _validate_rigid_trajectory("target trajectory", target_trajectory_np)
        if tool_trajectory_np.shape != target_trajectory_np.shape:
            raise ValueError(
                f"TACO tool/target frame mismatch for {index}: "
                f"{tool_trajectory_np.shape} vs {target_trajectory_np.shape}"
            )

        data_path, hand_poses, hand_translations, mano_joints = self._load_hand(record)
        if len(hand_poses) != len(tool_trajectory_np):
            raise ValueError(
                f"TACO hand/object frame mismatch for {index}: {len(hand_poses)} vs {len(tool_trajectory_np)}"
            )

        role = "tool" if self.side == "right" else "target"
        selected_trajectory_np = tool_trajectory_np if role == "tool" else target_trajectory_np
        hand_object_distance = torch.linalg.norm(
            hand_translations - torch.as_tensor(selected_trajectory_np[:, :3, 3], device=self.device), dim=-1
        ).median().item()
        if hand_object_distance > self.max_hand_object_distance:
            raise ValueError(
                f"TACO sequence {index} is inconsistent: median {self.side}-hand/{role} distance is "
                f"{hand_object_distance:.3f} m. Exclude or repair this annotation before training."
            )

        tool_mesh = _load_meter_mesh(record.tool_mesh_path)
        target_mesh = _load_meter_mesh(record.target_mesh_path)
        normalization = self._normalization_transform(
            tool_trajectory_np, target_trajectory_np, tool_mesh, target_mesh
        )
        effective_transform = self._base_mujoco2gym_transf @ self._source_frame_correction @ normalization

        selected_mesh = tool_mesh if role == "tool" else target_mesh
        source_object_id = record.tool_id if role == "tool" else record.target_id
        selected_object_id = self.new_obj if self.new_obj is not None else source_object_id
        selected_trajectory = torch.as_tensor(selected_trajectory_np, dtype=torch.float32, device=self.device)

        mesh_path, coacd_mesh_path, urdf_path = taco_asset_paths(self.data_dir, selected_object_id)
        missing_assets = [
            path for path in (mesh_path, coacd_mesh_path, urdf_path) if not os.path.isfile(path)
        ]
        if missing_assets:
            raise FileNotFoundError(
                f"Missing processed TACO assets: {missing_assets}. Run "
                f"`python scripts/preprocess_taco_assets.py --data-dir {self.data_dir} "
                f"--object-id {selected_object_id}` first."
            )
        if self.new_obj is not None:
            # The original meshes above intentionally define the shared scene
            # normalization. Only the right-hand tool geometry is replaced;
            # all hand and object trajectories remain unchanged.
            selected_mesh = _load_mesh(mesh_path)

        mano_joints = _linear_resample(mano_joints, TACO_SOURCE_FPS, MANIPTRANS_FPS)
        wrist_pos = mano_joints[:, 0]
        wrist_pos = wrist_pos - (mano_joints[:, 9] - wrist_pos) * 0.25
        wrist_pos = wrist_pos + torch.as_tensor(
            self.dexhand.relative_translation, dtype=wrist_pos.dtype, device=self.device
        )
        wrist_rot = aa_to_rotmat(hand_poses[:, :3])
        wrist_rot = _rotation_resample(wrist_rot, TACO_SOURCE_FPS, MANIPTRANS_FPS)
        wrist_rot = wrist_rot @ torch.as_tensor(
            self.dexhand.relative_rotation, dtype=wrist_rot.dtype, device=self.device
        )
        selected_trajectory = _pose_resample(selected_trajectory, TACO_SOURCE_FPS, MANIPTRANS_FPS)
        mano_joints_dict = self._mano_joint_dict(mano_joints)

        mesh = Meshes(
            verts=torch.as_tensor(np.asarray(selected_mesh.vertices)[None], dtype=torch.float32),
            faces=torch.as_tensor(np.asarray(selected_mesh.faces)[None], dtype=torch.int64),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": data_path,
            "sequence_id": index,
            "obj_id": (
                f"taco_override_{selected_object_id}"
                if self.new_obj is not None
                else f"taco_{selected_object_id}"
            ),
            "obj_mesh_path": mesh_path,
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": urdf_path,
            "obj_trajectory": selected_trajectory,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints_dict,
        }

        original_transform = self.mujoco2gym_transf
        try:
            self.mujoco2gym_transf = effective_transform
            self.process_data(data, index, rs_verts_obj)
        finally:
            self.mujoco2gym_transf = original_transform

        opt_path = os.path.join(
            "data",
            "retargeting",
            "TACO",
            f"mano2{str(self.dexhand)}",
            f"{index}.pkl",
        )
        self.load_retargeted_data(data, opt_path)
        return data


@register_manipdata("taco_rh")
class TacoDatasetDexHandRH(TacoDatasetDexHand):
    side = "right"


@register_manipdata("taco_lh")
class TacoDatasetDexHandLH(TacoDatasetDexHand):
    side = "left"
