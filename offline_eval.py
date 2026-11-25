from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader

CKPT_BASE = Path("/inspire/hdd/project/robot-reasoning/czxs253130583/openpi_r1lite_checkpoints/pi0_r1lite/r1lite-full-lora-20261119")
ACTION_KEYS = ("actions", "action")


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if "bfloat16" in str(value.dtype):
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def episode_bounds(dataset):
    index = getattr(dataset, "episode_data_index", None)
    if isinstance(index, dict) and "from" in index and "to" in index:
        pairs = zip(as_numpy(index["from"]), as_numpy(index["to"]))
        return [(int(a), int(b)) for a, b in pairs if int(b) > int(a)]

    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes:
        rows = episodes.values() if isinstance(episodes, dict) else episodes
        bounds = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            start = row.get("dataset_from_index", row.get("from", row.get("start_index")))
            end = row.get("dataset_to_index", row.get("to", row.get("end_index")))
            if start is not None and end is not None and int(end) > int(start):
                bounds.append((int(start), int(end)))
        if bounds:
            return sorted(bounds)

    return [(0, len(dataset))]


def select_indices(dataset, bounds, num_frames, val_fraction):
    if not bounds:
        bounds = [(0, len(dataset))]

    first_val_episode = int(len(bounds) * (1.0 - val_fraction))
    val_bounds = bounds[first_val_episode:] or [bounds[-1]]
    per_episode = max(1, num_frames // len(val_bounds))

    indices = []
    for start, end in val_bounds:
        lo = max(0, int(start) + 1)
        hi = min(len(dataset), int(end) - 1)
        if hi <= lo:
            continue
        take = min(per_episode, hi - lo)
        indices.extend(int(i) for i in np.linspace(lo, hi - 1, take, dtype=np.int64))

    indices = sorted(set(indices))
    if len(indices) > num_frames:
        keep = np.linspace(0, len(indices) - 1, num_frames, dtype=np.int64)
        indices = [indices[i] for i in keep]
    return indices, val_bounds


def action_key(sample):
    for key in ACTION_KEYS:
        if key in sample:
            return key
    raise KeyError(f"No action key found. sample keys={sorted(sample)}")


def fix_shape(value):
    value = as_numpy(value).astype(np.float32)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 1:
        value = value[None, :]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_r1lite")
    parser.add_argument("--ckpt-step", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(
        data_config,
        cfg.model.action_horizon,
        cfg.model,
    )
    bounds = episode_bounds(dataset)
    indices, val_bounds = select_indices(
        dataset, bounds, args.num_frames, args.val_fraction
    )

    print(f"config={args.config}")
    print(f"dataset_len={len(dataset)}")
    print(f"episodes={len(bounds)}")
    print(f"validation_episodes={len(val_bounds)}")
    print(f"selected_frames={len(indices)}")
    print("WARNING: 如果所有 episode 都参与过训练，这不是严格留出验证，只是训练分布回放。")

    sample = dataset[indices[0]]
    print("sample_keys:")
    for key in sorted(sample):
        value = sample[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        print(f"  {key}: shape={shape} dtype={getattr(value, 'dtype', type(value).__name__)}")

    if args.inspect:
        return

    if args.ckpt_step is None:
        raise SystemExit("需要指定 --ckpt-step")

    checkpoint = CKPT_BASE / str(args.ckpt_step)
    print(f"checkpoint={checkpoint}")
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    print("POLICY_LOADED", type(policy).__name__)

    records = []
    latencies = []

    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        key = action_key(sample)
        ground_truth = fix_shape(sample[key])

        observation = {
            name: value
            for name, value in sample.items()
            if name not in ("action", "actions", "action_is_pad")
        }
        output = policy.infer(observation)
        prediction = fix_shape(output["actions"])

        horizon = min(prediction.shape[0], ground_truth.shape[0])
        if prediction.shape[1] != ground_truth.shape[1]:
            raise ValueError(
                f"action dim mismatch: prediction={prediction.shape}, "
                f"ground_truth={ground_truth.shape}"
            )

        difference = prediction[:horizon] - ground_truth[:horizon]
        record = {
            "index": index,
            "action_mae": float(np.mean(np.abs(difference))),
            "action_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "per_dim_mae": np.mean(np.abs(difference), axis=0).tolist(),
            "finite": bool(np.isfinite(prediction).all()),
        }
        if prediction.shape[1] >= 3:
            record["translation_l2_raw"] = float(
                np.mean(np.linalg.norm(difference[:, :3], axis=-1))
            )
        if prediction.shape[1] >= 6:
            record["rotation_l2_raw"] = float(
                np.mean(np.linalg.norm(difference[:, 3:6], axis=-1))
            )
        if prediction.shape[1] >= 7:
            record["gripper_mae"] = float(
                np.mean(np.abs(difference[:, -1]))
            )

        timing = output.get("policy_timing", {})
        if timing.get("infer_ms") is not None:
            latencies.append(float(timing["infer_ms"]))
            record["infer_ms"] = latencies[-1]

        records.append(record)
        if count % 10 == 0 or count == len(indices):
            print(
                f"{count}/{len(indices)} "
                f"mae={record['action_mae']:.6f} "
                f"rmse={record['action_rmse']:.6f}"
            )

    def mean(name):
        values = [r[name] for r in records if name in r and np.isfinite(r[name])]
        return float(np.mean(values)) if values else None

    result = {
        "config": args.config,
        "checkpoint_step": args.ckpt_step,
        "num_frames": len(records),
        "action_dim": int(fix_shape(dataset[indices[0]][action_key(dataset[indices[0]])]).shape[1]),
        "mean_action_mae": mean("action_mae"),
        "mean_action_rmse": mean("action_rmse"),
        "mean_translation_l2_raw": mean("translation_l2_raw"),
        "mean_rotation_l2_raw": mean("rotation_l2_raw"),
        "mean_gripper_mae": mean("gripper_mae"),
        "mean_infer_ms": float(np.mean(latencies)) if latencies else None,
        "all_finite": bool(all(r["finite"] for r in records)),
        "per_dim_mae": np.mean(
            np.asarray([r["per_dim_mae"] for r in records], dtype=np.float32),
            axis=0,
        ).tolist(),
        "records": records,
    }

    output_dir = CKPT_BASE / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"offline_eval_{args.ckpt_step}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("=== RESULT ===")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "records"},
        ensure_ascii=False,
        indent=2,
    ))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
