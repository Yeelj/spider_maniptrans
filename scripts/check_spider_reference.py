"""Check a SPIDER bundle and real training targets without running PPO."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/retargeting_spider_native")
    parser.add_argument("--dataset", choices=("TACO", "OakInk-v2"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--new-obj")
    parser.add_argument("--object-reference", choices=("demonstration", "spider_rollout"), default="demonstration")
    parser.add_argument("--episode-length", type=int, default=1500)
    parser.add_argument("--control-dt", type=float, default=1 / 60)
    parser.add_argument("--reject-failed", action="store_true")
    parser.add_argument("--asset-root")
    parser.add_argument("--asset-path-map", default="{}", help="JSON mapping of old absolute roots to new roots")
    parser.add_argument("--check-models", action="store_true")
    parser.add_argument("--simulator-check", action="store_true", help="Also check static PhysX frames and done/reset on cuda:0")
    parser.add_argument("--rh-checkpoint", default="assets/imitator_rh_artimano.pth")
    parser.add_argument("--lh-checkpoint", default="assets/imitator_lh_artimano.pth")
    parser.add_argument("--report")
    args = parser.parse_args()
    # Isaac Gym requires this import before torch, even for these CPU probes.
    import isaacgym
    import torch
    from main.dataset.factory import ManipDataFactory
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    from main.spider_reference_check import check_fk, check_models, probe_reference

    torch.set_num_threads(4)
    result = {"task": args.task, "dataset": args.dataset, "object_reference": args.object_reference, "hands": {}}
    observations = []
    for side, side_name in (("rh", "right"), ("lh", "left")):
        hand = DexHandFactory.create_hand("artimano", side_name)
        dataset = ManipDataFactory.create_data("spider", side_name,
            reference_root=args.root, reference_dataset=args.dataset, new_obj=args.new_obj,
            object_reference=args.object_reference, allow_failed_reference=not args.reject_failed,
            asset_root=args.asset_root, asset_path_map=json.loads(args.asset_path_map),
            max_seq_len=args.episode_length, device="cpu", dexhand=hand, control_dt=args.control_dt)
        data = dataset[args.task]
        observations.append(probe_reference(data, hand))
        probe_reference(data, hand, future_length=4)
        result["hands"][side] = {"manifest": data["manifest_path"], "frames": len(data["wrist_pos"]),
            "source_object_id": data["obj_id"], "selected_asset_id": data["selected_asset_id"],
            "validation_status": data["validation_status"], "fk_max_error_m": check_fk(data, hand),
            "target_reward_observation_checks": "passed", "future_boundary_checks": "passed"}
    if args.check_models:
        result["models"] = check_models(observations, args.rh_checkpoint, args.lh_checkpoint)
    else:
        result["models"] = "not_checked (use --check-models)"
    if args.simulator_check:
        from main.spider_reference_check import check_simulator
        result["simulator"] = check_simulator(args)
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.report:
        Path(args.report).write_text(output + "\n")


if __name__ == "__main__":
    main()
