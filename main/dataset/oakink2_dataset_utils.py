import trimesh
import os
from pathlib import Path
from typing import List


def oakink2_object_paths(data_dir, object_id: str):
    """Resolve a native OakInk-V2 mesh and its prepared collision/URDF assets."""
    if not isinstance(object_id, str) or not object_id or Path(object_id).name != object_id:
        raise ValueError(f"Expected an OakInk-V2 object ID such as 'O02@0015@00004', got {object_id!r}")
    if object_id in (".", ".."):
        raise ValueError(f"Invalid OakInk-V2 object ID: {object_id!r}")
    data_dir = Path(data_dir)
    object_dir = data_dir / "object_preview" / "align_ds" / object_id
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Missing OakInk-V2 object directory: {object_dir}")
    meshes = sorted(path for path in object_dir.iterdir() if path.is_file() and path.suffix in (".obj", ".ply"))
    if len(meshes) != 1:
        raise ValueError(f"Expected one .obj or .ply mesh in {object_dir}, found {meshes}")

    mesh_path = meshes[0]
    coacd_path = data_dir / "coacd_object_preview" / "align_ds" / object_id / mesh_path.name
    urdf_path = coacd_path.with_suffix(".urdf")
    missing = [str(path) for path in (coacd_path, urdf_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed OakInk-V2 assets: {missing}. Prepare the CoACD mesh and matching URDF "
            "as described in README.md under OakInk-V2 prerequisites."
        )
    return str(mesh_path), str(urdf_path)


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) != 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values())
        )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_obj_map(obj_prefix: str, obj_list: List[str]):
    res = {}
    for obj_id in obj_list:
        obj_filedir = os.path.join(obj_prefix, obj_id)
        candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
        assert len(candidate_list) == 1
        obj_filename = candidate_list[0]
        obj_filepath = os.path.join(obj_filedir, obj_filename)
        if os.path.splitext(obj_filename)[-1] == ".obj":
            mesh = as_mesh(trimesh.load_mesh(obj_filepath, process=False, skip_materials=True, force="mesh"))
        else:
            mesh = trimesh.load(obj_filepath, process=False, force="mesh")
        res[obj_id] = (mesh, obj_filepath)
    return res


oakink2_obj_scale = {
    "O02@0206@00001": 1.15,
    "O02@0206@00002": 0.95,
    "O02@0029@00011": 0.9,
    "O02@0029@00012": 1.2,
    "O02@0015@00020": 0.98,
    "O02@0015@00019": 1.02,
}
oakink2_obj_mass = {
    "O02@0015@00002": 0.101,
    "O02@0015@00001": 0.027,
    "C12001": 0.114,
    "O02@0030@00002": 0.0144,
    "O02@0033@00002": 0.014,
    "O02@0011@00003": 0.12,
    "O02@0206@00002": 0.163,
}  # TODO: add more objects
