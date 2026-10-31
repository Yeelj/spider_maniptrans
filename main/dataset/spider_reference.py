"""Robot references already expressed in ManipTrans world coordinates."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh

from .base import ManipData
from .decorators import register_manipdata
from .transform import aa_to_rotmat
from lib.utils.spider_reference import (
    find_manifest, name_order, normalize_tool, object_transform, read_bundle,
    resolve_asset, sha256,
)


class SpiderReference(ManipData):
    side = None

    def __init__(self, *, reference_root, reference_dataset, dexhand, new_obj=None,
                 object_reference="demonstration", allow_failed_reference=True,
                 asset_root=None, asset_path_map=None, max_seq_len=int(1e10),
                 device="cuda:0", control_dt=1 / 60, **kwargs):
        # Do not run the legacy constructor/process_data: these frames have
        # already been aligned and need neither MANO nor a CUDA chamfer module.
        self.data_dir = str(Path(reference_root).resolve())
        self.reference_dataset = reference_dataset
        self.tool = normalize_tool(new_obj, reference_dataset)
        if object_reference not in ("demonstration", "spider_rollout"):
            raise ValueError("Invalid objectReference: " + str(object_reference))
        self.object_reference = object_reference
        self.allow_failed_reference = bool(allow_failed_reference)
        self.asset_root = asset_root
        self.asset_path_map = dict(asset_path_map or {})
        self.dexhand = dexhand
        self.device = device
        self.max_seq_len = max_seq_len
        self.control_dt = float(control_dt)
        self.skip = 1
        self.dataset_fps = self.output_fps = 60.0
        self.data_pathes = list((Path(self.data_dir) / self.tool / reference_dataset / "training").glob("*/manifest.json"))

    @lru_cache(maxsize=32)
    def __getitem__(self, task):
        path = find_manifest(self.data_dir, self.tool, self.reference_dataset, task)
        manifest, arrays = read_bundle(path, self.allow_failed_reference)
        if (manifest["source_dataset"] != self.reference_dataset
                or normalize_tool(manifest["new_obj"], self.reference_dataset) != self.tool
                or str(task) not in [manifest["task"]] + manifest["task_aliases"]):
            raise ValueError("Requested dataset/tool/task disagrees with SPIDER manifest")
        if manifest["recorded_frames"] > self.max_seq_len:
            raise ValueError("SPIDER trajectory has {} frames; raise task.env.episodeLength to at least {} (no implicit truncation)".format(manifest["recorded_frames"], manifest["recorded_frames"]))
        self.output_fps = self.dataset_fps = float(manifest["fps"])
        if not np.isclose(self.control_dt, 1 / self.output_fps, rtol=1e-5, atol=1e-8):
            raise ValueError("sim.dt * controlFrequencyInv must equal 1 / SPIDER fps")
        side = self.side
        info = manifest[side]
        if self.dexhand.name != manifest["robot_type"] or self.dexhand.side != side:
            raise ValueError("Training hand differs from the SPIDER robot")
        dof_order = name_order(info["dof_names"], list(self.dexhand.dof_names), side + " DOF")
        body_order = name_order(info["body_names"], list(self.dexhand.body_names), side + " link")
        if self.dexhand.body_names[0] != "palm":
            raise ValueError("Training hand must have palm first")
        if sha256(self.dexhand.urdf_path) != info["hand_urdf_sha256"]:
            raise ValueError("Training hand URDF checksum differs from exported reference")
        paths = {key: resolve_asset(info[key], path, self.asset_root, self.asset_path_map)
                 for key in ("obj_mesh_path", "obj_urdf_path")}
        for key in ("obj_mesh", "obj_urdf"):
            if sha256(paths[key + "_path"]) != info[key + "_sha256"]:
                raise ValueError("SPIDER asset checksum differs: " + key)

        def tensor(value):
            return torch.tensor(np.asarray(value).copy(), dtype=torch.float32, device=self.device)

        mesh = trimesh.load(paths["obj_mesh_path"], process=False, force="mesh")
        # Deterministic uniform surface samples without touching training RNGs.
        rng = np.random.RandomState(0)
        areas = np.asarray(mesh.area_faces)
        if not np.isfinite(areas).all() or areas.sum() <= 0:
            raise ValueError("Invalid SPIDER object surface: " + paths["obj_mesh_path"])
        triangles = mesh.triangles[rng.choice(len(areas), 1000, p=areas / areas.sum())]
        uv = rng.rand(1000, 2)
        reflected = uv.sum(axis=1) > 1
        uv[reflected] = 1 - uv[reflected]
        vertices = triangles[:, 0] + (triangles[:, 1] - triangles[:, 0]) * uv[:, :1] + (triangles[:, 2] - triangles[:, 0]) * uv[:, 1:]
        # TACO PLY is already in metres. Only the manifest actor scale applies.
        vertices = tensor(vertices * info["object_scale"])
        wrist = tensor(arrays[side + "_wrist_pos"])
        rotation = tensor(arrays[side + "_wrist_rotvec"])
        joints = tensor(arrays[side + "_joint_pos"][:, body_order])
        dofs = tensor(arrays[side + "_dof_pos"][:, dof_order])
        obj = tensor(object_transform(arrays, side, self.object_reference))
        dt = 1 / self.output_fps
        wrist_vel = self.compute_velocity(wrist[:, None], dt).squeeze(1)
        wrist_ang_vel = self.compute_angular_velocity(aa_to_rotmat(rotation[:, None]), dt).squeeze(1)
        joint_vel = self.compute_velocity(joints, dt)
        data = {
            "reference_source": "spider", "manifest_path": str(path),
            "reference_task": manifest["task"], "reference_fps": self.output_fps,
            "reference_trajectory_sha256": manifest["trajectory_sha256"],
            "object_reference": self.object_reference, "validation_status": manifest["validation_status"],
            "data_path": info["reference_pkl"], "obj_id": info["source_object_id"],
            "selected_asset_id": info["selected_asset_id"], "scene_objs": [],
            "obj_verts": vertices, "obj_trajectory": obj,
            "object_scale": info["object_scale"], "object_mass_override": info["object_mass_override"],
            "object_density": info["object_density"], "object_friction": info["object_friction"],
            "object_rolling_friction": info["object_rolling_friction"],
            "object_torsion_friction": info["object_torsion_friction"],
            "wrist_pos": wrist, "opt_wrist_pos": wrist,
            "wrist_rot": rotation, "opt_wrist_rot": rotation,
            "opt_dof_pos": dofs, "opt_dof_velocity": self.compute_dof_velocity(dofs, dt),
            "reference_joint_pos": joints, "reference_joint_velocity": joint_vel,
            "wrist_velocity": wrist_vel, "opt_wrist_velocity": wrist_vel,
            "wrist_angular_velocity": wrist_ang_vel, "opt_wrist_angular_velocity": wrist_ang_vel,
            "obj_velocity": self.compute_velocity(obj[:, None, :3, 3], dt).squeeze(1),
            "obj_angular_velocity": self.compute_angular_velocity(obj[:, None, :3, :3], dt).squeeze(1),
            **paths,
        }
        tip_indices = [self.dexhand.body_names.index(finger + "_tip")
                       for finger in ("thumb", "index", "middle", "ring", "pinky")]
        # Match legacy nearest-surface-point distance, bounded memory for full
        # 1294-frame references; all inputs use the selected geometry and clock.
        distances = []
        for start in range(0, len(obj), 64):
            transform = obj[start:start + 64]
            world_vertices = vertices[None] @ transform[:, :3, :3].transpose(-1, -2) + transform[:, None, :3, 3]
            distances.append(torch.cdist(joints[start:start + 64, tip_indices], world_vertices).amin(dim=-1))
        data["tips_distance"] = torch.cat(distances)
        print("SPIDER reference " + json.dumps({"manifest": str(path), "side": side,
              "frames": len(obj), "fps": self.output_fps, "objectReference": self.object_reference,
              "source_object_id": data["obj_id"], "selected_asset_id": data["selected_asset_id"],
              "validation_status": manifest["validation_status"], "validation_passed": manifest["validation_passed"]}))
        return data


@register_manipdata("spider_rh")
class SpiderReferenceRH(SpiderReference):
    side = "rh"


@register_manipdata("spider_lh")
class SpiderReferenceLH(SpiderReference):
    side = "lh"
