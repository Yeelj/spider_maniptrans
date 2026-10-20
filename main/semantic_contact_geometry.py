"""Deterministic collision geometry for checkpoint evaluation.

URDFCollisionKinematics and its sampling helpers are copied unchanged from
PhysGraph main/dataset/contact_trajectory_retarget.py. Keeping this subset
outside main.dataset avoids dataset auto-registration and retargeting imports.
No optimizer or trajectory correction is imported or executed here.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _as_mesh(scene_or_mesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = tuple(scene_or_mesh.geometry.values())
        if not geometries:
            raise ValueError("Mesh scene contains no geometry")
        return trimesh.util.concatenate(geometries)
    if not isinstance(scene_or_mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a trimesh mesh, got {type(scene_or_mesh)!r}")
    return scene_or_mesh


def _origin_matrix(origin: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.size != 3 or rpy.size != 3:
        raise ValueError(f"Invalid URDF origin: {origin.attrib}")
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _deterministic_mesh_samples(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    """Area-weighted deterministic surface samples, independent of RNG state."""

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if len(triangles) == 0:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            return np.empty((0, 3), dtype=np.float64)
        indices = np.linspace(0, len(vertices) - 1, count).astype(np.int64)
        return vertices[indices]
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        indices = np.linspace(0, len(triangles) - 1, count).astype(np.int64)
    else:
        cumulative = np.cumsum(areas)
        targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cumulative[-1]
        indices = np.searchsorted(cumulative, targets, side="left")
    selected = triangles[np.clip(indices, 0, len(triangles) - 1)]
    sample_ids = np.arange(count, dtype=np.float64)
    u = np.mod(sample_ids * 0.6180339887498949 + 0.2113248654051871, 1.0)
    v = np.mod(sample_ids * 0.4142135623730950 + 0.3660254037844386, 1.0)
    reflect = (u + v) > 1.0
    u[reflect] = 1.0 - u[reflect]
    v[reflect] = 1.0 - v[reflect]
    return selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (
        selected[:, 2] - selected[:, 0]
    )


def _primitive_samples(geometry: ET.Element, count: int) -> np.ndarray:
    box = geometry.find("box")
    if box is not None:
        extents = np.fromstring(box.attrib["size"], sep=" ", dtype=np.float64)
        return _deterministic_mesh_samples(trimesh.creation.box(extents=extents), count)
    sphere = geometry.find("sphere")
    if sphere is not None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(sphere.attrib["radius"]))
        return _deterministic_mesh_samples(mesh, count)
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        mesh = trimesh.creation.cylinder(
            radius=float(cylinder.attrib["radius"]), height=float(cylinder.attrib["length"]), sections=24
        )
        return _deterministic_mesh_samples(mesh, count)
    return np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True)
class URDFJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    mimic_joint: Optional[str]
    mimic_multiplier: float
    mimic_offset: float


class URDFCollisionKinematics:
    """Minimal URDF FK plus deterministic samples from every collision link."""

    def __init__(
        self,
        urdf_path: Path,
        dof_names: Sequence[str],
        points_per_link: int = 32,
    ):
        self.urdf_path = urdf_path.resolve()
        self.dof_names = list(dof_names)
        root = ET.parse(self.urdf_path).getroot()
        links = [element.attrib["name"] for element in root.findall("link")]
        self.joints: List[URDFJoint] = []
        child_links = set()
        for element in root.findall("joint"):
            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]
            axis_element = element.find("axis")
            axis = np.fromstring(
                "0 0 1" if axis_element is None else axis_element.attrib.get("xyz", "0 0 1"),
                sep=" ",
                dtype=np.float64,
            )
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm
            mimic_element = element.find("mimic")
            self.joints.append(
                URDFJoint(
                    name=element.attrib["name"],
                    parent=parent,
                    child=child,
                    joint_type=element.attrib.get("type", "fixed"),
                    origin=_origin_matrix(element.find("origin")),
                    axis=axis,
                    mimic_joint=None if mimic_element is None else mimic_element.attrib["joint"],
                    mimic_multiplier=1.0
                    if mimic_element is None
                    else float(mimic_element.attrib.get("multiplier", 1.0)),
                    mimic_offset=0.0
                    if mimic_element is None
                    else float(mimic_element.attrib.get("offset", 0.0)),
                )
            )
            child_links.add(child)
        roots = [link for link in links if link not in child_links]
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link, found {roots}")
        self.root_link = roots[0]
        self.children: Dict[str, List[URDFJoint]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)

        collision_points: Dict[str, List[np.ndarray]] = {}
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            collision_elements = link.findall("collision")
            if not collision_elements:
                continue
            per_collision = max(4, int(np.ceil(points_per_link / len(collision_elements))))
            for collision in collision_elements:
                geometry = collision.find("geometry")
                if geometry is None:
                    continue
                mesh_element = geometry.find("mesh")
                if mesh_element is not None:
                    mesh_path = self._resolve_mesh_path(mesh_element.attrib["filename"])
                    mesh = _as_mesh(trimesh.load(mesh_path, process=False, force="mesh"))
                    scale = np.fromstring(mesh_element.attrib.get("scale", "1 1 1"), sep=" ")
                    if scale.size != 3:
                        raise ValueError(f"Invalid mesh scale in {self.urdf_path}: {mesh_element.attrib}")
                    mesh = mesh.copy()
                    mesh.vertices = np.asarray(mesh.vertices) * scale[None]
                    points = _deterministic_mesh_samples(mesh, per_collision)
                else:
                    points = _primitive_samples(geometry, per_collision)
                if len(points):
                    points = _transform_points(points, _origin_matrix(collision.find("origin")))
                    collision_points.setdefault(link_name, []).append(points)
        self.link_points: Dict[str, np.ndarray] = {
            link: np.concatenate(parts, axis=0)[:points_per_link]
            for link, parts in collision_points.items()
        }
        if not self.link_points:
            raise ValueError(f"No collision geometry was sampled from {self.urdf_path}")
        self.point_link_names: List[str] = []
        self.point_link_ids: List[int] = []
        points = []
        for link_id, (link, link_points) in enumerate(self.link_points.items()):
            self.point_link_names.append(link)
            self.point_link_ids.extend([link_id] * len(link_points))
            points.append(link_points)
        self.local_points = np.concatenate(points, axis=0)
        self.point_link_ids_array = np.asarray(self.point_link_ids, dtype=np.int64)

    def _resolve_mesh_path(self, filename: str) -> Path:
        if filename.startswith("package://"):
            filename = filename[len("package://") :]
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = self.urdf_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"URDF collision mesh does not exist: {candidate}")
        return candidate

    def _joint_value(self, joint: URDFJoint, q_map: Mapping[str, float]) -> float:
        # Isaac Gym exposes mimic joints as regular DOFs for the Inspire URDF,
        # and the project stores those values explicitly in opt_dof_pos.
        if joint.name in q_map:
            return q_map[joint.name]
        if joint.mimic_joint is not None:
            return q_map.get(joint.mimic_joint, 0.0) * joint.mimic_multiplier + joint.mimic_offset
        return q_map.get(joint.name, 0.0)

    def link_transforms(self, dof: np.ndarray) -> Dict[str, np.ndarray]:
        q_map = {name: float(value) for name, value in zip(self.dof_names, dof)}
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                motion = np.eye(4, dtype=np.float64)
                value = self._joint_value(joint, q_map)
                if joint.joint_type in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(joint.axis * value).as_matrix()
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis * value
                transforms[joint.child] = transforms[parent] @ joint.origin @ motion
                stack.append(joint.child)
        return transforms

    def root_frame_surface_points(self, dof: np.ndarray) -> np.ndarray:
        transforms = self.link_transforms(dof)
        result = []
        for link in self.point_link_names:
            if link not in transforms:
                raise ValueError(f"Collision link {link!r} is disconnected in {self.urdf_path}")
            result.append(_transform_points(self.link_points[link], transforms[link]))
        return np.concatenate(result, axis=0)

    def world_surface_trajectory(
        self,
        wrist_pos: np.ndarray,
        wrist_rotvec: np.ndarray,
        dof: np.ndarray,
        frame_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if frame_ids is None:
            frame_ids = np.arange(len(wrist_pos), dtype=np.int64)
        wrist_rotations = Rotation.from_rotvec(wrist_rotvec[frame_ids]).as_matrix()
        frames = []
        for local_idx, frame_id in enumerate(frame_ids):
            root_points = self.root_frame_surface_points(dof[frame_id])
            frames.append(root_points @ wrist_rotations[local_idx].T + wrist_pos[frame_id])
        return np.stack(frames, axis=0)
