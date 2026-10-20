"""Read-only, policy-independent semantic contact evaluation (no Isaac Gym import)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from collections.abc import Mapping
import json
import math
import pickle

import torch


@dataclass(frozen=True)
class SemanticContactEvalConfig:
    enabled: bool = False
    forceThreshold: float = 1.0
    contactDistance: float = 0.01
    pointsPerLink: int = 32
    envChunkSize: int = 16
    minRedContactRatio: float = 0.90
    maxBlackContactRatio: float = 0.05
    maxBlackViolationFrameRatio: float = 0.05
    minContactFrames: int = 3
    minPlannedSemanticContactCoverage: float = 0.80
    expectedContactDistance: float = 0.03
    contactRequiredMaskFile: str = None
    contactRetargetRoot: str = "data/retargeting_contact"

    @classmethod
    def from_mapping(cls, config):
        result = cls(**{f.name: config[f.name] for f in fields(cls) if f.name in config})
        for name in ("forceThreshold", "contactDistance", "expectedContactDistance"):
            value = getattr(result, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"semanticContactEval.{name} must be finite and nonnegative")
        for name in ("minRedContactRatio", "maxBlackContactRatio", "maxBlackViolationFrameRatio",
                     "minPlannedSemanticContactCoverage"):
            if not 0 <= getattr(result, name) <= 1:
                raise ValueError(f"semanticContactEval.{name} must be in [0, 1]")
        for name in ("pointsPerLink", "envChunkSize", "minContactFrames"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"semanticContactEval.{name} must be a positive integer")
        return result


def full_fragment_end(right_lengths, left_lengths, episode_length):
    """Inclusive reference endpoint for this repository's t + 1 + 3 rule."""
    return (torch.minimum(right_lengths, left_lengths).clamp(max=episode_length) - 4).clamp_min(0)


def _boolean_mask(values, label):
    try:
        mask = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} must be a one-dimensional boolean array") from error
    if mask.ndim != 1 or (mask.numel() and mask.dtype != torch.bool):
        raise ValueError(f"{label} must be a one-dimensional boolean array")
    return mask.bool().cpu().clone()


def _required_interval(values, label):
    try:
        interval = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} must contain two integer frame indices") from error
    if (interval.shape != (2,) or interval.dtype == torch.bool
            or interval.is_floating_point() or interval.is_complex()):
        raise ValueError(f"{label} must contain two integer frame indices")
    start, end = interval.tolist()
    if start < 0 or end < start:
        raise ValueError(f"{label} must satisfy 0 <= start <= end (inclusive)")
    return start, end


def contact_required_mask_from_metadata(metadata, planned_frames):
    """Map source intent from the grasp interval's LOCAL axis onto the plan.

    Supports both full corrected PKLs and their contact_retarget metadata.
    A missing annotation returns None; a malformed annotation raises.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("Contact metadata must be a mapping")
    metadata = metadata.get("contact_retarget", metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError("contact_retarget must be a mapping")
    intent = metadata.get("source_contact_intent")
    if intent is not None and not isinstance(intent, Mapping):
        raise ValueError("source_contact_intent must be a mapping")
    has_source_mask = intent is not None and "contact_frame_mask" in intent
    if "grasp_interval" not in metadata:
        if has_source_mask:
            raise ValueError("Source contact_frame_mask requires its grasp_interval offset")
        return None
    start, end = _required_interval(metadata["grasp_interval"], "grasp_interval")
    mask = torch.zeros(planned_frames, dtype=torch.bool)
    stop = min(end + 1, planned_frames)
    if has_source_mask:
        local = _boolean_mask(intent["contact_frame_mask"], "source contact_frame_mask")
        if len(local) != end - start + 1:
            raise ValueError("Source contact_frame_mask must cover the local grasp_interval")
        if start < stop:
            mask[start:stop] = local[:stop - start]
    elif start < stop:
        mask[start:stop] = True
    return mask


def canonical_contact_metadata_path(root, dataset_name, hand_name, data_idx, new_obj):
    """Find evaluation metadata without loading/replacing policy trajectories."""
    if new_obj is None or str(new_obj) == "":
        return None
    tool = str(new_obj)
    if dataset_name == "TACO" and tool.isdigit():
        tool = tool.zfill(3)
    components = [dataset_name, f"mano2{hand_name}", str(data_idx), tool]
    for value in components:
        if (not value or Path(value).is_absolute() or len(Path(value).parts) != 1
                or value in (".", "..")):
            raise ValueError(f"Invalid contact metadata path component: {value!r}")
    directory = Path(root).joinpath(*components)
    # TACO uses data_idx.pkl; OakInk's original filename includes more context.
    matches = sorted(directory.glob("*.pkl"))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous canonical contact metadata: {matches}")
    return matches[0] if matches else None


def load_contact_required_masks(data_indices, planned_frames, tips_distances, config,
                                metadata_paths=None):
    """Load one immutable plan mask per environment, returning masks + sources.

    A configured JSON is authoritative: missing IDs/short arrays are errors,
    never silently replaced with per-method reference inference.
    metadata_paths maps full data IDs to common corrected PKLs (annotation only).
    """
    shared = None
    if config.contactRequiredMaskFile is not None:
        with Path(config.contactRequiredMaskFile).open(encoding="utf-8") as stream:
            shared = json.load(stream)
        if not isinstance(shared, dict):
            raise ValueError("contactRequiredMaskFile must contain a data-ID mapping")
    lengths = torch.as_tensor(planned_frames).cpu().long().tolist()
    if len(lengths) != len(data_indices) or not lengths or min(lengths) < 1:
        raise ValueError("Each environment must have one positive full plan length")
    masks, sources, cache = [], [], {}
    metadata_paths = metadata_paths or {}
    for i, (data_idx, length) in enumerate(zip(data_indices, lengths)):
        data_idx = str(data_idx)
        cache_key = (data_idx, length)
        if cache_key in cache:
            mask, source = cache[cache_key]
        elif shared is not None:
            if data_idx not in shared:
                raise ValueError(f"Shared contactRequiredMaskFile missing dataIndices={data_idx}")
            entry = shared[data_idx]
            if isinstance(entry, dict):
                if set(entry) != {"required_intervals"} or not isinstance(entry["required_intervals"], list):
                    raise ValueError(f"Invalid required_intervals entry for {data_idx}")
                mask = torch.zeros(length, dtype=torch.bool)
                for interval in entry["required_intervals"]:
                    start, end = _required_interval(interval, f"{data_idx} required_intervals")
                    mask[start:min(end + 1, length)] = True
            else:
                mask = _boolean_mask(entry, f"{data_idx} shared contact mask")
                if len(mask) < length:
                    raise ValueError(f"Shared contact mask for {data_idx} must cover all {length} planned frames")
                mask = mask[:length]
            source = f"shared JSON {config.contactRequiredMaskFile}"
        else:
            mask, source = None, None
            path = metadata_paths.get(data_idx)
            if path is not None:
                with Path(path).open("rb") as stream:
                    metadata = pickle.load(stream)
                mask = contact_required_mask_from_metadata(metadata, length)
                if mask is not None:
                    payload = metadata.get("contact_retarget", metadata)
                    intent = payload.get("source_contact_intent") or {}
                    kind = "source_contact_intent.contact_frame_mask" if "contact_frame_mask" in intent else "grasp_interval (coarse)"
                    source = f"{path}: {kind}"
            if mask is None:
                if tips_distances is None:
                    raise ValueError(f"No contact annotation or reference tips_distance for {data_idx}")
                distances = torch.as_tensor(tips_distances[i]).detach().cpu()
                if (distances.ndim != 2 or len(distances) < length or distances.shape[1] < 1
                        or not torch.isfinite(distances[:length]).all() or (distances[:length] < 0).any()):
                    raise ValueError(f"Invalid reference tips_distance for {data_idx}")
                mask = (distances[:length] <= config.expectedContactDistance).any(dim=-1)
                source = f"FALLBACK checkpoint reference tips_distance <= {config.expectedContactDistance}"
        if cache_key not in cache:
            cache[cache_key] = mask, source
            print(f"[semanticContactEval] dataIndices={data_idx} contact_required: {source}; "
                  f"required_frames={int(mask.sum())}/{length}")
        masks.append(mask)
        sources.append(source)
    return masks, sources


def load_semantic_regions(mesh_path, point_count=512):
    """Same deterministic 512 red/black vertex samples as PhysGraph datasets.

    Mesh coordinates must already be in meters in the physical tool frame.
    Actor scale is applied separately. Never substitute policy soft features.
    """
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a semantic triangle mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.visual.vertex_colors)
    if (mesh.visual.kind != "vertex" or colors.ndim != 2
            or colors.shape[0] != len(vertices) or colors.shape[1] < 3
            or not np.isfinite(vertices).all()):
        raise ValueError(f"Missing or invalid per-vertex red/black semantics: {mesh_path}")
    rgb = colors[:, :3].astype(np.int16, copy=False)
    red = (rgb[:, 0] >= 200) & (rgb[:, 1] <= 40) & (rgb[:, 2] <= 40)
    black = np.max(rgb, axis=1) <= 40
    if not red.any() or not black.any() or (~(red | black)).any():
        raise ValueError(
            f"Tool must be completely labeled red/black: {mesh_path}; "
            f"red={red.sum()}, black={black.sum()}, unclassified={(~(red | black)).sum()}"
        )
    return tuple(torch.from_numpy(np.asarray(
        vertices[mask][np.linspace(0, mask.sum() - 1, point_count, dtype=np.int64)],
        dtype=np.float32,
    )) for mask in (red, black))


def quat_rotate_xyzw(quaternion, points):
    """Rotate vectors with Isaac Gym's unit quaternion convention."""
    xyz = quaternion[..., :3]
    uv = torch.cross(xyz, points, dim=-1)
    return points + 2 * (quaternion[..., 3:4] * uv + torch.cross(xyz, uv, dim=-1))


class CollisionSurfaceProxy:
    """One event = one force-gated collision sample at one control step."""

    def __init__(self, local_points, body_indices, regions, region_ids, scales, config):
        self.config = config
        self.local_points = local_points
        self.body_indices = body_indices.to(device=local_points.device, dtype=torch.long)
        self.regions = [(r.to(local_points), b.to(local_points)) for r, b in regions]
        self.region_ids = region_ids.to(device=local_points.device, dtype=torch.long)
        self.scales = scales.to(local_points)
        if self.local_points.shape != (len(self.body_indices), 3):
            raise ValueError("Each collision sample must have one rigid body index")
        if (self.scales.ndim != 1 or self.region_ids.shape != self.scales.shape
                or not self.regions or (self.region_ids < 0).any()
                or (self.region_ids >= len(self.regions)).any()):
            raise ValueError("Each environment must have one tool scale and valid semantic region id")
        if not torch.isfinite(self.local_points).all() or (self.body_indices < 0).any():
            raise ValueError("Collision samples and rigid body indices must be valid")
        for red, black in self.regions:
            for points in (red, black):
                if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                        or not torch.isfinite(points).all()):
                    raise ValueError("Semantic regions must be nonempty finite [N, 3] point sets")
        if not torch.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("Tool scales must be finite and positive")

    @classmethod
    def from_urdf(cls, urdf_path, dof_names, body_handles, regions, region_ids, scales, config):
        from main.semantic_contact_geometry import URDFCollisionKinematics

        model = URDFCollisionKinematics(Path(urdf_path), dof_names, config.pointsPerLink)
        missing = [name for name in model.point_link_names if body_handles.get(name, -1) < 0]
        if missing:
            raise ValueError(f"Collision links missing from simulator: {missing}")
        indices = [body_handles[model.point_link_names[i]] for i in model.point_link_ids_array]
        device = scales.device
        return cls(torch.as_tensor(model.local_points, device=device, dtype=torch.float32),
                   torch.tensor(indices, device=device), regions, region_ids, scales, config)

    @torch.no_grad()
    def detect(self, rigid_body_states, net_contact_forces, tool_states, active):
        num_envs = len(tool_states)
        red_count = torch.zeros(num_envs, dtype=torch.long, device=tool_states.device)
        black_count = torch.zeros_like(red_count)
        for region_id, (red, black) in enumerate(self.regions):
            ids = (active & (self.region_ids == region_id)).nonzero(as_tuple=False).flatten()
            for chunk in ids.split(self.config.envChunkSize):
                if chunk.numel() == 0:
                    continue
                bodies = rigid_body_states[chunk][:, self.body_indices]
                local = self.local_points.unsqueeze(0).expand(len(chunk), -1, -1)
                world = quat_rotate_xyzw(bodies[..., 3:7], local) + bodies[..., :3]
                tool = tool_states[chunk]
                inverse = tool[:, 3:7].clone()
                inverse[:, :3] *= -1
                tool_local = quat_rotate_xyzw(
                    inverse[:, None].expand(-1, len(self.local_points), -1),
                    world - tool[:, None, :3],
                )
                scale = self.scales[chunk, None, None]
                # Direct cdist avoids cancellation near a 1e-6 semantic boundary.
                red_distance = torch.cdist(tool_local, red[None] * scale,
                    compute_mode="donot_use_mm_for_euclid_dist").amin(dim=-1)
                black_distance = torch.cdist(tool_local, black[None] * scale,
                    compute_mode="donot_use_mm_for_euclid_dist").amin(dim=-1)
                force = net_contact_forces[chunk][:, self.body_indices].norm(dim=-1)
                valid = ((force >= self.config.forceThreshold) & torch.isfinite(force)
                         & (torch.minimum(red_distance, black_distance) <= self.config.contactDistance))
                is_black = black_distance <= red_distance + 1e-6
                red_count[chunk] = (valid & ~is_black).sum(dim=-1)
                black_count[chunk] = (valid & is_black).sum(dim=-1)
        return red_count, black_count


class SemanticContactAccumulator:
    """Per-environment state, independent of reward, progress offset and policy."""

    def __init__(self, end_frames, config, contact_required):
        self.end_frames = end_frames.clone().long()
        self.config = config
        self.planned_frames = self.end_frames + 1
        if (self.end_frames < 0).any() or len(contact_required) != len(self.end_frames):
            raise ValueError("Each environment needs an endpoint and a full contact-required mask")
        self.contact_required = torch.zeros(
            (len(self.end_frames), int(self.planned_frames.max().item())),
            dtype=torch.bool, device=self.end_frames.device,
        )
        for i, length in enumerate(self.planned_frames.tolist()):
            mask = _boolean_mask(contact_required[i], "contact_required")
            if len(mask) != length:
                raise ValueError("contact_required must match the complete plan length")
            self.contact_required[i, :length] = mask.to(self.end_frames.device)
        self.planned_required_frames = self.contact_required.sum(dim=1)
        self.red_events = torch.zeros_like(self.end_frames)
        self.black_events = torch.zeros_like(self.end_frames)
        self.eval_frames = torch.zeros_like(self.end_frames)
        self.black_frames = torch.zeros_like(self.end_frames)
        self.contact_frames = torch.zeros_like(self.end_frames)
        self.executed_required_frames = torch.zeros_like(self.end_frames)
        self.compliant_required_frames = torch.zeros_like(self.end_frames)
        self.black_required_frames = torch.zeros_like(self.end_frames)
        self.missing_contact_frames = torch.zeros_like(self.end_frames)
        self.visited = torch.zeros_like(self.contact_required)
        self.last_frame = torch.full_like(self.end_frames, -1)
        self.continuous = torch.zeros_like(self.end_frames, dtype=torch.bool)
        self.finished = torch.ones_like(self.continuous)

    def reset(self, env_ids, start_frames, eligible=True):
        for value in (self.red_events, self.black_events, self.eval_frames,
                      self.black_frames, self.contact_frames, self.executed_required_frames,
                      self.compliant_required_frames, self.black_required_frames, self.missing_contact_frames):
            value[env_ids] = 0
        self.visited[env_ids] = False
        self.last_frame[env_ids] = start_frames - 1
        self.continuous[env_ids] = (start_frames == 0) & eligible
        self.finished[env_ids] = False

    def active(self, frames):
        indices = frames.clamp(0, self.contact_required.shape[1] - 1).long()
        visited = self.visited.gather(1, indices[:, None]).squeeze(1)
        return (~self.finished & (frames >= 0) & (frames <= self.end_frames)
                & ~visited)

    @torch.no_grad()
    def record(self, frames, red_events, black_events):
        active = self.active(frames)
        # Repeated callbacks for the same reference frame cannot double count.
        self.continuous &= ~active | (frames == self.last_frame + 1)
        self.continuous &= self.finished | (frames >= self.last_frame)
        self.red_events += torch.where(active, red_events, 0)
        self.black_events += torch.where(active, black_events, 0)
        self.eval_frames += active.long()
        self.black_frames += (active & (black_events > 0)).long()
        self.contact_frames += (active & (red_events + black_events > 0)).long()
        indices = frames.clamp(0, self.contact_required.shape[1] - 1).long()
        required = active & self.contact_required.gather(1, indices[:, None]).squeeze(1)
        has_red, has_black = red_events > 0, black_events > 0
        self.executed_required_frames += required.long()
        self.compliant_required_frames += (required & has_red & ~has_black).long()
        self.black_required_frames += (required & has_black).long()
        self.missing_contact_frames += (required & ~has_red & ~has_black).long()
        ids = active.nonzero(as_tuple=False).flatten()
        self.visited[ids, indices[ids]] = True
        self.last_frame[active] = frames[active]

    @torch.no_grad()
    def finish(self, done, task_success):
        ids = (done.bool() & ~self.finished).nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return None
        complete = (self.continuous & (self.last_frame == self.end_frames)
                    & (self.eval_frames == self.end_frames + 1))
        total = self.red_events + self.black_events
        red_ratio = self.red_events.double() / total.clamp_min(1)
        black_ratio = self.black_events.double() / total.clamp_min(1)
        violation_ratio = self.black_frames.double() / self.eval_frames.clamp_min(1)
        coverage = self.compliant_required_frames.double() / self.planned_required_frames.clamp_min(1)
        semantic_success = (
            task_success.bool() & complete & (total > 0)
            & (self.contact_frames >= self.config.minContactFrames)
            & (red_ratio >= self.config.minRedContactRatio)
            & (black_ratio <= self.config.maxBlackContactRatio)
            & (violation_ratio <= self.config.maxBlackViolationFrameRatio)
            & (self.planned_required_frames > 0)
            & (coverage >= self.config.minPlannedSemanticContactCoverage)
        )
        snapshot = {name: getattr(self, name)[ids].clone() for name in (
            "red_events", "black_events", "eval_frames", "black_frames", "contact_frames",
            "planned_frames", "planned_required_frames", "executed_required_frames",
            "compliant_required_frames", "black_required_frames", "missing_contact_frames")}
        snapshot["unexecuted_required_frames"] = (
            self.planned_required_frames[ids] - self.executed_required_frames[ids]
        ).clone()
        snapshot.update(env_ids=ids.clone(), task_success=task_success[ids].bool().clone(),
                        complete=complete[ids].clone(), semantic_success=semantic_success[ids].clone())
        self.finished[ids] = True
        return snapshot


class SemanticContactSummary:
    """Batch ratios use summed event/frame counts, never episode-ratio means."""

    def __init__(self):
        self.total_episodes = 0
        self.total_successes = 0

    def add(self, snapshot, done_ids):
        ids = snapshot["env_ids"]
        done_ids = done_ids.flatten().to(ids.device)
        if len(ids) != len(done_ids) or not torch.equal(ids.sort().values, done_ids.sort().values):
            raise ValueError("Semantic terminal snapshot must match exactly the ended environments")
        count = len(ids)
        if count == 0:
            raise ValueError("Cannot summarize an empty terminal batch")
        nr = int(snapshot["red_events"].sum().item())
        nb = int(snapshot["black_events"].sum().item())
        frames = int(snapshot["eval_frames"].sum().item())
        black_frames = int(snapshot["black_frames"].sum().item())
        planned = int(snapshot["planned_frames"].sum().item())
        required = int(snapshot["planned_required_frames"].sum().item())
        compliant = int(snapshot["compliant_required_frames"].sum().item())
        successes = int(snapshot["semantic_success"].sum().item())
        self.total_episodes += count
        self.total_successes += successes
        return {
            "operation_completion_ratio": frames / planned,
            "planned_semantic_contact_coverage": compliant / required if required else None,
            "red_contact_ratio": nr / (nr + nb) if nr + nb else None,
            "black_contact_ratio": nb / (nr + nb) if nr + nb else None,
            "black_violation_frame_ratio": black_frames / frames if frames else None,
            "semantic_task_success_rate": successes / count,
            "semantic_sr_total": self.total_successes / self.total_episodes,
            "eval_frames": frames,
            "planned_frames": planned,
            "required_frames": required,
            "compliant_required_frames": compliant,
            "black_required_frames": int(snapshot["black_required_frames"].sum().item()),
            "missing_contact_frames": int(snapshot["missing_contact_frames"].sum().item()),
            "unexecuted_required_frames": int(snapshot["unexecuted_required_frames"].sum().item()),
            "complete_episodes": int(snapshot["complete"].sum().item()),
            "episodes": count,
            "no_contact_episodes": int(((snapshot["red_events"] + snapshot["black_events"]) == 0).sum().item()),
        }


def format_semantic_contact_summary(summary):
    ratios = ("operation_completion_ratio", "planned_semantic_contact_coverage", "black_violation_frame_ratio",
              "semantic_task_success_rate", "semantic_sr_total")
    parts = [f"{name}: {summary[name]:.4f}" if summary[name] is not None else f"{name}: NA"
             for name in ratios]
    parts.extend((f"eval_frames: {summary['eval_frames']}",
                  f"complete_episodes: {summary['complete_episodes']}/{summary['episodes']}",
                  f"no_contact_episodes: {summary['no_contact_episodes']}"))
    for name in ("planned_frames", "required_frames", "compliant_required_frames",
                 "missing_contact_frames", "unexecuted_required_frames"):
        parts.append(f"{name}: {summary[name]}")
    # Event ratios remain auxiliary diagnostics after the four main metrics.
    for name in ("red_contact_ratio", "black_contact_ratio"):
        parts.append(f"{name}: {summary[name]:.4f}" if summary[name] is not None else f"{name}: NA")
    return " ".join(parts)
