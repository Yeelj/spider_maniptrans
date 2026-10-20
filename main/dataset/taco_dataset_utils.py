import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple


TACO_SEQUENCE_ID_PATTERN = re.compile(r"^[a-z0-9]{2}_[a-z0-9_]+_[0-9]{7}$")


@dataclass(frozen=True)
class TacoSequenceRecord:
    sequence_id: str
    triplet: str
    sequence_name: str
    hand_pose_dir: str
    object_pose_dir: str
    tool_id: str
    target_id: str
    tool_pose_path: str
    target_pose_path: str
    tool_mesh_path: str
    target_mesh_path: str


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def parse_taco_triplet(triplet: str) -> Tuple[str, str, str]:
    """Parse TACO's unquoted directory form, e.g. ``(brush, brush, bowl)``."""
    if not (triplet.startswith("(") and triplet.endswith(")")):
        raise ValueError(f"Invalid TACO triplet directory name: {triplet}")
    fields = tuple(part.strip() for part in triplet[1:-1].split(","))
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError(f"Invalid TACO triplet directory name: {triplet}")
    return fields


def make_taco_sequence_id(triplet: str, sequence_name: str) -> str:
    """Create the globally unique compact ID used by ManipTrans.

    The convention requested for this integration is::

        (brush, brush, bowl)/20230919_036 -> bb_bowl_0919036

    The first two letters are the action/tool initials, followed by the full
    target slug and ``MMDD + sequence number``. TACO V1 sequence names are
    globally unique; the triplet prefix also makes IDs human-readable.
    """
    action, tool, target = parse_taco_triplet(triplet)
    action_slug = _slug(action)
    tool_slug = _slug(tool)
    target_slug = _slug(target)
    if not action_slug or not tool_slug or not target_slug:
        raise ValueError(f"Cannot build an ID from TACO triplet: {triplet}")
    if re.fullmatch(r"20[0-9]{6}_[0-9]{3}", sequence_name) is None:
        raise ValueError(f"Invalid TACO sequence name: {sequence_name}")
    compact_sequence = sequence_name[4:].replace("_", "")
    return f"{action_slug[0]}{tool_slug[0]}_{target_slug}_{compact_sequence}"


def is_taco_sequence_id(index) -> bool:
    return isinstance(index, str) and TACO_SEQUENCE_ID_PATTERN.fullmatch(index) is not None


def _find_role_pose(object_pose_dir: str, role: str) -> Tuple[str, str]:
    prefix = f"{role}_"
    candidates = sorted(
        filename
        for filename in os.listdir(object_pose_dir)
        if filename.startswith(prefix) and filename.endswith(".npy")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {role} pose in {object_pose_dir}, found {candidates}"
        )
    filename = candidates[0]
    object_id = filename[len(prefix) : -len(".npy")]
    if re.fullmatch(r"[0-9]{3}", object_id) is None:
        raise ValueError(f"Invalid TACO object ID in {filename}: {object_id!r}")
    return object_id, os.path.join(object_pose_dir, filename)


@lru_cache(maxsize=None)
def build_taco_sequence_index(data_dir: str) -> Dict[str, TacoSequenceRecord]:
    data_dir = os.path.abspath(data_dir)
    hand_root = os.path.join(data_dir, "Hand_Poses")
    object_pose_root = os.path.join(data_dir, "Object_Poses")
    object_model_root = os.path.join(data_dir, "object_models_released")
    for required_dir in (hand_root, object_pose_root, object_model_root):
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Required TACO directory does not exist: {required_dir}")

    records: Dict[str, TacoSequenceRecord] = {}
    for triplet in sorted(os.listdir(hand_root)):
        triplet_hand_dir = os.path.join(hand_root, triplet)
        if not os.path.isdir(triplet_hand_dir):
            continue
        parse_taco_triplet(triplet)
        for sequence_name in sorted(os.listdir(triplet_hand_dir)):
            hand_pose_dir = os.path.join(triplet_hand_dir, sequence_name)
            if not os.path.isdir(hand_pose_dir):
                continue
            sequence_id = make_taco_sequence_id(triplet, sequence_name)
            object_pose_dir = os.path.join(object_pose_root, triplet, sequence_name)
            if not os.path.isdir(object_pose_dir):
                raise FileNotFoundError(f"Missing TACO object poses: {object_pose_dir}")

            tool_id, tool_pose_path = _find_role_pose(object_pose_dir, "tool")
            target_id, target_pose_path = _find_role_pose(object_pose_dir, "target")
            tool_mesh_path = os.path.join(object_model_root, f"{tool_id}_cm.obj")
            target_mesh_path = os.path.join(object_model_root, f"{target_id}_cm.obj")
            for mesh_path in (tool_mesh_path, target_mesh_path):
                if not os.path.isfile(mesh_path):
                    raise FileNotFoundError(f"Missing TACO object model: {mesh_path}")

            record = TacoSequenceRecord(
                sequence_id=sequence_id,
                triplet=triplet,
                sequence_name=sequence_name,
                hand_pose_dir=hand_pose_dir,
                object_pose_dir=object_pose_dir,
                tool_id=tool_id,
                target_id=target_id,
                tool_pose_path=tool_pose_path,
                target_pose_path=target_pose_path,
                tool_mesh_path=tool_mesh_path,
                target_mesh_path=target_mesh_path,
            )
            if sequence_id in records:
                other = records[sequence_id]
                raise ValueError(
                    f"Duplicate TACO sequence ID {sequence_id}: "
                    f"{other.triplet}/{other.sequence_name} and {triplet}/{sequence_name}"
                )
            records[sequence_id] = record

    if not records:
        raise RuntimeError(f"No TACO sequences found under {hand_root}")
    return records


def taco_asset_paths(data_dir: str, object_id: str) -> Tuple[str, str, str]:
    if re.fullmatch(r"[0-9]{3}", object_id) is None:
        raise ValueError(f"Invalid TACO object ID: {object_id!r}")
    object_dir = os.path.join(data_dir, "object_models", object_id)
    coacd_dir = os.path.join(data_dir, "coacd_object_models", object_id)
    mesh_path = os.path.join(object_dir, f"{object_id}.ply")
    coacd_mesh_path = os.path.join(coacd_dir, f"{object_id}.ply")
    urdf_path = os.path.join(coacd_dir, f"{object_id}.urdf")
    return mesh_path, coacd_mesh_path, urdf_path
