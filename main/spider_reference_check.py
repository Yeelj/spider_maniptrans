"""Reference probes using the real environment packing, observation and reward hooks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def probe_reference(data, hand, future_length=1, device="cpu"):
    from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import DexHandManipBiHEnv
    from .dataset.transform import aa_to_quat, rotmat_to_quat

    count = len(data["wrist_pos"])
    frames = torch.tensor([0, max(count - 2, 0), count - 1, count - 1], device=device)
    n = len(frames)
    probe = SimpleNamespace(
        device=device, num_envs=n, max_episode_length=count,
        dexhand_rh=hand, dexhand_lh=hand, reference_source="spider",
        obs_future_length=future_length, training=False, headless=True,
        progress_buf=frames.clone(), running_progress_buf=torch.zeros(n, dtype=torch.long, device=device),
        _obs_keys=["q", "cos_q", "sin_q", "base_state"],
        _privileged_obs_keys=["dq", "manip_obj_pos", "manip_obj_quat", "manip_obj_vel", "manip_obj_ang_vel", "tip_force", "manip_obj_com", "manip_obj_weight"],
        _prop_dump_info={}, rollout_len=None,
        reset_buf=torch.zeros(n, dtype=torch.long, device=device),
        actions=torch.zeros(n, 56, device=device), total_rew_buf=torch.zeros(n, device=device),
        dof_force=torch.zeros(n, 44, device=device),
        net_cf=torch.zeros(n, 28, 3, device=device),
        apply_forces=torch.zeros(n, 28, 3, device=device), apply_torque=torch.zeros(n, 28, 3, device=device),
        gym=SimpleNamespace(get_sim_params=lambda sim: SimpleNamespace(gravity=SimpleNamespace(z=-9.81))), sim=None,
    )
    # These poisoned human fields must never enter robot target packing.
    poisoned = dict(data, mano_joints={"wrong_human": torch.full((count, 3), 999.0, device=device)},
                    mano_joints_velocity={"wrong_human": torch.full((count, 3), 999.0, device=device)})
    packed = DexHandManipBiHEnv.pack_data(probe, [poisoned] * n, side="rh")
    base = torch.cat([data["wrist_pos"][frames], aa_to_quat(data["wrist_rot"][frames])[:, [1, 2, 3, 0]],
                      data["wrist_velocity"][frames], data["wrist_angular_velocity"][frames]], dim=-1)
    joints = torch.zeros(n, 28, 13, device=device)
    joints[:, :, :3] = data["reference_joint_pos"][frames]
    joints[:, :, 7:10] = data["reference_joint_velocity"][frames]
    obj = data["obj_trajectory"][frames]
    q = data["opt_dof_pos"][frames]
    states = {"base_state": base, "joints_state": joints, "q": q, "cos_q": q.cos(), "sin_q": q.sin(),
              "dq": data["opt_dof_velocity"][frames], "manip_obj_pos": obj[:, :3, 3],
              "manip_obj_quat": rotmat_to_quat(obj[:, :3, :3])[:, [1, 2, 3, 0]],
              "manip_obj_vel": data["obj_velocity"][frames], "manip_obj_ang_vel": data["obj_angular_velocity"][frames]}
    for side in ("rh", "lh"):
        setattr(probe, "demo_data_" + side, packed)
        setattr(probe, side + "_states", states)
        setattr(probe, "_manip_obj_" + side + "_root_state", torch.cat([
            states["manip_obj_pos"], states["manip_obj_quat"], states["manip_obj_vel"], states["manip_obj_ang_vel"]], dim=-1))
        setattr(probe, "dexhand_" + side + "_handles", {name: i for i, name in enumerate(hand.body_names)})
        setattr(probe, side + "_tips_contact_history", torch.ones(n, 3, 5, dtype=torch.bool, device=device))
        setattr(probe, "manip_obj_" + side + "_com", torch.zeros(n, 3, device=device))
        setattr(probe, "manip_obj_" + side + "_mass", torch.ones(n, device=device) * 0.1)
        setattr(probe, "obj_bps_" + side, torch.zeros(n, 128, device=device))
    reward = DexHandManipBiHEnv.compute_reward_side(probe, probe.actions, "rh")
    for key, value in zip(("obj_pos_cm", "obj_rot_degrees", "joint_cm", "tip_cm"), reward[-4:]):
        tolerance = 0.1 if key == "obj_rot_degrees" else 1e-4
        if value.abs().max().item() > tolerance:
            raise ValueError("Perfect target state has nonzero {}: {}".format(key, value))
    # SPIDER evaluates frame T-1; legacy's three-frame termination margin is
    # independently covered by the regression suite.
    if reward[2].tolist() != [False, False, True, True]:
        raise ValueError("SPIDER termination did not include the real final frame")
    probe.reference_source = "legacy"
    legacy_reward = DexHandManipBiHEnv.compute_reward_side(probe, probe.actions, "rh")
    if not torch.equal(legacy_reward[2], frames + 4 >= count):
        raise ValueError("Legacy three-frame termination margin changed")
    probe.reference_source = "spider"
    probe.progress_buf[-1] += 5  # exercise overshoot and every future offset
    observations = DexHandManipBiHEnv.compute_observations_side(probe, "rh")
    indices = torch.minimum(probe.progress_buf[:, None] + 1 + torch.arange(future_length, device=device), packed["seq_len"][:, None] - 1)
    expected_wrist = packed["wrist_pos"][torch.arange(n, device=device)[:, None], indices] - base[:, None, :3]
    torch.testing.assert_close(observations["target"][:, :3 * future_length], expected_wrist.reshape(n, -1))
    expected_links = packed["mano_joints"][torch.arange(n, device=device)[:, None], indices].reshape(n, future_length, 27, 3) - joints[:, None, 1:, :3]
    start, width = 23 * future_length, 81 * future_length
    torch.testing.assert_close(observations["target"][:, start:start + width], expected_links.reshape(n, -1))
    if not all(torch.isfinite(value).all() for value in observations.values()):
        raise ValueError("Reference observations are not finite")
    return observations


def check_fk(data, hand):
    from .semantic_contact_geometry import URDFCollisionKinematics

    model = URDFCollisionKinematics(Path(hand.urdf_path), hand.dof_names, 1)
    errors = []
    for frame in (0, len(data["wrist_pos"]) // 2, len(data["wrist_pos"]) - 1):
        transforms = model.link_transforms(data["opt_dof_pos"][frame].cpu().numpy())
        local = np.stack([transforms[name][:3, 3] for name in hand.body_names])
        rot = Rotation.from_rotvec(data["wrist_rot"][frame].cpu().numpy()).as_matrix()
        world = local @ rot.T + data["wrist_pos"][frame].cpu().numpy()
        errors.append(float(np.linalg.norm(world - data["reference_joint_pos"][frame].cpu().numpy(), axis=-1).max()))
    if max(errors) > 1e-4:
        raise ValueError("Hand/link FK differs from SPIDER reference: {} m".format(errors))
    return max(errors)


def check_models(observations, rh_checkpoint, lh_checkpoint):
    """Build the unmodified residual/base networks, including strict weights."""
    import lib  # registers the existing Hydra resolvers
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from gym import spaces
    from rl_games.algos_torch.model_builder import ModelBuilder, register_model, register_network
    from lib.rl.models import ModelA2CContinuousLogStd
    from lib.rl.res_models import ModelA2CContinuousLogStdResBiH
    from lib.rl.network_builder import DictObsBuilder
    from lib.rl.network_builder_residual_bih import ResBiHDictObsBuilder

    for path in (rh_checkpoint, lh_checkpoint):
        if not path or not Path(path).is_file():
            raise FileNotFoundError("Pass both existing Artimano base checkpoints")
    with initialize_config_dir(config_dir=str(Path(__file__).parent / "cfg"), version_base="1.1"):
        cfg = compose(config_name="config", overrides=["task=ResDexHand", "side=BiH", "dexhand=artimano"])
    cfg.rh_base_model_checkpoint = str(Path(rh_checkpoint).resolve())
    cfg.lh_base_model_checkpoint = str(Path(lh_checkpoint).resolve())
    params = OmegaConf.to_container(cfg.rl_train.params, resolve=True)
    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_model("res_bih_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResBiH)
    register_network("dict_obs_actor_critic", DictObsBuilder)
    register_network("res_bih_dict_obs_actor_critic", ResBiHDictObsBuilder)
    obs = {key: torch.cat([observations[0][key], observations[1][key]], dim=-1).cpu() for key in observations[0]}
    shape = spaces.Dict({key: spaces.Box(-np.inf, np.inf, shape=(value.shape[-1],)) for key, value in obs.items()})
    model = ModelBuilder().load(params).build({"actions_num": 56, "input_shape": shape,
        "num_seqs": 4, "value_size": 1, "normalize_input": True, "normalize_value": True,
        "normalize_input_excluded_keys": [], "use_pid_control": False, **params})
    model.eval()
    with torch.no_grad():
        result = model({"is_train": False, "obs": obs, "rnn_states": None})
    if result["actions"].shape != (4, 56) or result["base_actions"].shape != (4, 56):
        raise ValueError("Artimano action dimensions differ")
    if not all(torch.isfinite(value).all() for value in result.values() if torch.is_tensor(value)):
        raise ValueError("Non-finite base/residual policy output")
    return {"strict_load": True, "base_actions_finite": True, "observation_shapes": {key: list(value.shape) for key, value in obs.items()}}


def check_simulator(args):
    """Static PhysX frame/FK and terminal-reset probe in an isolated simulator.

    Contacts are disabled only in this disposable static probe, so a single
    PhysX update computes articulation FK without contact-driven motion. PPO
    smoke runs use the normal collision settings.
    """
    import lib
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgym import gymtorch
    from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import DexHandManipBiHEnv

    overrides = ["task=ResDexHand", "side=BiH", "dexhand=artimano", "num_envs=2", "headless=true",
                 "randomStateInit=false", "rolloutStateInit=false", "task.task.randomize=false",
                 "task.env.referenceSource=spider", "task.env.datasetType=spider",
                 "task.env.referenceRoot=" + str(Path(args.root).resolve()),
                 "task.env.referenceDataset=" + args.dataset,
                 "task.env.objectReference=" + args.object_reference,
                 "task.env.episodeLength=" + str(args.episode_length),
                 "task.sim.dt=" + str(args.control_dt), "dataIndices=[" + args.task + "]"]
    if args.new_obj is not None:
        overrides.append("task.env.newObj=" + args.new_obj)
    with initialize_config_dir(config_dir=str(Path(__file__).parent / "cfg"), version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.task.env.allowFailedReference = not args.reject_failed
    cfg.task.env.assetRoot = args.asset_root
    import json
    cfg.task.env.assetPathMap = json.loads(args.asset_path_map)
    env = DexHandManipBiHEnv(OmegaConf.to_container(cfg.task, resolve=True),
                            sim_device="cuda:0", rl_device="cuda:0", headless=True, graphics_device_id=-1)
    report = []
    try:
        for handle in env.envs:
            for actor in range(env.gym.get_actor_count(handle)):
                props = env.gym.get_actor_rigid_shape_properties(handle, actor)
                for prop in props:
                    prop.filter = 1
                env.gym.set_actor_rigid_shape_properties(handle, actor, props)
        length = int(env.demo_data_rh["seq_len"][0])
        env_ids = torch.arange(env.num_envs, device=env.device)
        for frame in (0, length // 2, length - 1):
            env.rollout_begin = frame
            env._reset_default(env_ids)
            # This is a static pose check. Reset's normal trajectory velocities
            # remain covered by target probes and normal PPO rollouts.
            env._qd.zero_()
            env._rh_base_state[:, 7:].zero_()
            env._lh_base_state[:, 7:].zero_()
            env.apply_forces.zero_()
            env.apply_torque.zero_()
            env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_state))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env._refresh()
            errors = {}
            for side in ("rh", "lh"):
                demo = getattr(env, "demo_data_" + side)
                state = getattr(env, side + "_states")
                expected = torch.cat([demo["wrist_pos"][:, frame, None], demo["mano_joints"][:, frame].reshape(env.num_envs, 27, 3)], dim=1)
                error = torch.norm(state["joints_state"][:, :, :3] - expected, dim=-1).max().item()
                errors[side] = error
                if error > 0.002:
                    raise ValueError("Static PhysX hand FK differs at frame {} {}: {} m".format(frame, side, error))
            env.obs_future_length = 4
            env.compute_observations()
            env.compute_reward(env.actions)
            if not all(torch.isfinite(value).all() for value in env.obs_dict.values() if torch.is_tensor(value)):
                raise ValueError("Non-finite simulator observations")
            if frame == length - 1:
                if not env.reset_buf.bool().all():
                    raise ValueError("Final frame did not terminate")
                env.rollout_begin = None
                env.random_state_init = True
                _, reset_ids = env.reset_done()
                if len(reset_ids) != env.num_envs or not (env.progress_buf < length).all():
                    raise ValueError("done -> random reset failed")
            report.append({"frame": frame, "static_physx_max_link_error_m": errors})
        return report
    finally:
        env.gym.destroy_sim(env.sim)
