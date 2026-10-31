"""Portable SPIDER training bundle contract (NumPy/SciPy only, Python 3.8)."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_tool(value, dataset):
    if dataset not in ("TACO", "OakInk-v2"):
        raise ValueError("referenceDataset must be TACO or OakInk-v2")
    if value is None or value == "original":
        return "original"
    value = str(value)
    if dataset == "TACO":
        if not value.isdigit() or not 0 <= int(value) <= 999:
            raise ValueError("TACO newObj must be a three digit object ID")
        value = value.zfill(3)
    return path_component(value)


def path_component(value):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("Invalid reference path component: {!r}".format(value))
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def name_order(source, target, label):
    if (not source or len(set(source)) != len(source)
            or len(set(target)) != len(target) or set(source) != set(target)):
        raise ValueError("{} names must be unique and match the training hand: {} vs {}".format(label, source, target))
    return [source.index(name) for name in target]


def validate_transform(value, label):
    if not np.isfinite(value).all() or not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous transform: " + label)
    rot = value[..., :3, :3]
    if not np.allclose(rot @ np.swapaxes(rot, -1, -2), np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(rot), 1, atol=2e-4):
        raise ValueError("Invalid rotation matrix: " + label)


def validate_bundle(manifest, arrays, allow_failed=True):
    """Structural checks always apply, independently of quality policy."""
    m = manifest
    if m.get("schema_version") != 1 or m.get("robot_type") != "artimano":
        raise ValueError("Only schema 1 Artimano SPIDER bundles are supported")
    normalize_tool(m.get("new_obj"), m.get("source_dataset"))
    path_component(m["task"])
    if (m.get("frame_coordinates") != "maniptrans_world"
            or m.get("rotation_convention") != "rotvec" or m.get("state_quaternion") != "xyzw"):
        raise ValueError("Unsupported SPIDER coordinates or rotation convention")
    count = m["recorded_frames"]
    if type(count) is not int or count < 2 or count != m["expected_frames"] or m.get("complete") is not True:
        raise ValueError("SPIDER reference must contain every real execution frame")
    fps = float(m["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid reference fps")
    if not allow_failed and (m.get("validation_status") != "passed" or m.get("validation_passed") is False):
        raise ValueError("Reference quality failed; allowFailedReference is false")
    if m.get("validation_status") not in ("passed", "failed", "skipped"):
        raise ValueError("Missing reference validation status")
    if not np.array_equal(arrays["frame_index"], np.arange(count)):
        raise ValueError("Reference frame indices must be contiguous from zero")
    if arrays["time_seconds"].shape != (count,) or not np.allclose(arrays["time_seconds"], np.arange(count) / fps, atol=1e-7, rtol=0):
        raise ValueError("Reference timestamps do not match fps")
    for side in ("rh", "lh"):
        info = m[side]
        for key in ("dof_names", "body_names"):
            names = info[key]
            if not names or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
                raise ValueError("Duplicate or invalid {} {}".format(side, key))
        if "palm" not in info["body_names"]:
            raise ValueError("Missing palm link")
        shapes = {"wrist_pos": (3,), "wrist_rotvec": (3,),
                  "dof_pos": (len(info["dof_names"]),),
                  "joint_pos": (len(info["body_names"]), 3),
                  "source_object_transform": (4, 4), "sim_object_state": (13,)}
        for key, shape in shapes.items():
            value = arrays[side + "_" + key]
            if value.shape != (count,) + shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("Invalid shape or values in {}_{}".format(side, key))
        validate_transform(arrays[side + "_source_object_transform"], side + " object")
        quat = arrays[side + "_sim_object_state"][:, 3:7]
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=5e-3):
            raise ValueError("Invalid simulated object quaternion")
        palm = info["body_names"].index("palm")
        if not np.allclose(arrays[side + "_joint_pos"][:, palm], arrays[side + "_wrist_pos"], atol=1e-4):
            raise ValueError("Palm and wrist coordinates disagree")
        scale = float(info["object_scale"])
        mass = info["object_mass_override"]
        if not np.isfinite(scale) or scale <= 0 or (mass is not None and (not np.isfinite(mass) or mass <= 0)):
            raise ValueError("Invalid object scale/mass")
        for key in ("object_density", "object_friction", "object_rolling_friction", "object_torsion_friction"):
            value = float(info[key])
            if not np.isfinite(value) or value < 0 or (key == "object_density" and value == 0):
                raise ValueError("Invalid " + key)
        for key in ("source_object_id", "selected_asset_id", "obj_mesh_path", "obj_urdf_path", "hand_urdf", "hand_urdf_sha256"):
            if not isinstance(info.get(key), str) or not info[key]:
                raise ValueError("Missing {} {}".format(side, key))


def find_manifest(root, tool, dataset, task):
    task = path_component(str(task))
    base = Path(root).resolve() / normalize_tool(tool, dataset) / dataset / "training"
    direct = base / task / "manifest.json"
    if direct.is_file():
        return direct
    matches = []
    for path in sorted(base.glob("*/manifest.json")):
        with path.open() as stream:
            meta = json.load(stream)
        if task in meta.get("task_aliases", []):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError("Expected one explicit SPIDER task/alias {!r} under {}; found {}".format(task, base, matches))
    return matches[0]


@lru_cache(maxsize=16)
def _read_bundle(path, manifest_stamp, trajectory_stamp, allow_failed):
    with open(path) as stream:
        manifest = json.load(stream)
    trajectory = Path(path).parent / path_component(manifest["trajectory_file"])
    if sha256(trajectory) != manifest["trajectory_sha256"]:
        raise ValueError("SPIDER trajectory checksum differs from manifest")
    with np.load(str(trajectory), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_bundle(manifest, arrays, allow_failed)
    for value in arrays.values():
        value.setflags(write=False)
    return manifest, arrays


def read_bundle(path, allow_failed=True):
    path = Path(path).resolve()
    with path.open() as stream:
        header = json.load(stream)
    trajectory = path.parent / path_component(header["trajectory_file"])
    return _read_bundle(str(path), path.stat().st_mtime_ns, trajectory.stat().st_mtime_ns, bool(allow_failed))


def resolve_asset(value, manifest_path, asset_root=None, path_map=None):
    path = Path(value)
    for old, new in sorted((path_map or {}).items(), key=lambda pair: -len(pair[0])):
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        path = Path(new) / relative
        break
    if not path.is_absolute():
        path = Path(asset_root or Path(manifest_path).parent) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("Missing SPIDER asset {}; configure assetRoot/assetPathMap explicitly".format(path))
    return str(path)


def object_transform(arrays, side, policy):
    if policy == "demonstration":
        return arrays[side + "_source_object_transform"]
    if policy != "spider_rollout":
        raise ValueError("objectReference must be demonstration or spider_rollout")
    state = arrays[side + "_sim_object_state"]
    transform = np.repeat(np.eye(4)[None], len(state), axis=0)
    transform[:, :3, :3] = Rotation.from_quat(state[:, 3:7]).as_matrix()
    transform[:, :3, 3] = state[:, :3]
    return transform


def prepare_training_config(cfg):
    """Normalize Hydra paths and fail before allocating a large simulator."""
    if cfg.task.env.get("referenceSource", "legacy") != "spider":
        return []
    from hydra.utils import to_absolute_path

    env = cfg.task.env
    if (cfg.side != "BiH" or cfg.bimanual_mode != "united" or cfg.dexhand != "artimano"
            or cfg.task.name != "ResDexHandBiH" or env.datasetType != "spider"):
        raise ValueError("SPIDER training requires ResDexHand, side=BiH, bimanual_mode=united, dexhand=artimano, datasetType=spider")
    # The existing frozen imitators consume exactly one future frame. Their
    # target slicing/architecture is intentionally unchanged in this work.
    if env.obsFutureLength != 1:
        raise ValueError("Frozen base policies require obsFutureLength=1; multi-future environment indexing is supported, but these checkpoints use one frame")
    if env.objectReference not in ("demonstration", "spider_rollout"):
        raise ValueError("Invalid objectReference")
    env.referenceRoot = to_absolute_path(env.referenceRoot)
    env.newObj = None if normalize_tool(env.newObj, env.referenceDataset) == "original" else normalize_tool(env.newObj, env.referenceDataset)
    if env.assetRoot:
        env.assetRoot = to_absolute_path(env.assetRoot)
    env.assetPathMap = {str(old): to_absolute_path(new) for old, new in env.assetPathMap.items()}
    for side in ("rh", "lh"):
        key = side + "_base_model_checkpoint"
        if not cfg.get(key) or not Path(to_absolute_path(cfg[key])).is_file():
            raise FileNotFoundError("SPIDER requires an existing Artimano base policy: " + key)
        cfg[key] = to_absolute_path(cfg[key])
    reports = []
    for task in env.dataIndices:
        path = find_manifest(env.referenceRoot, env.newObj, env.referenceDataset, task)
        manifest, arrays = read_bundle(path, env.allowFailedReference)
        if manifest["source_dataset"] != env.referenceDataset or normalize_tool(manifest["new_obj"], env.referenceDataset) != normalize_tool(env.newObj, env.referenceDataset):
            raise ValueError("Requested SPIDER namespace differs from manifest")
        if manifest["recorded_frames"] > env.episodeLength:
            raise ValueError("episodeLength={} would truncate {} frames; increase it".format(env.episodeLength, manifest["recorded_frames"]))
        if not np.isclose(cfg.task.sim.dt * env.get("controlFrequencyInv", 1), 1 / manifest["fps"], rtol=1e-5, atol=1e-8):
            raise ValueError("Simulation control clock differs from SPIDER reference fps")
        reports.append({"manifest": str(path), "task": manifest["task"], "new_obj": manifest["new_obj"],
                        "frames": manifest["recorded_frames"], "fps": manifest["fps"],
                        "object_reference": env.objectReference, "state_source": manifest["state_source"],
                        "validation_status": manifest["validation_status"], "validation_passed": manifest["validation_passed"],
                        "failed_at": manifest["failed_at"], "failure_reason": manifest["failure_reason"],
                        "trajectory_sha256": manifest["trajectory_sha256"],
                        "assets": {side: manifest[side] for side in ("rh", "lh")}})
    if not reports:
        raise ValueError("SPIDER training requires at least one dataIndices task")
    batch = int(env.numEnvs) * int(cfg.rl_train.params.config.horizon_length)
    if batch % int(cfg.rl_train.params.config.minibatch_size):
        raise ValueError("num_envs * horizon_length must be divisible by minibatch_size")
    return reports
