from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if "bfloat16" in str(value.dtype):
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def episode_bounds(dataset):
    index = getattr(dataset, "episode_data_index", None)
    if isinstance(index, dict) and "from" in index and "to" in index:
        starts = to_numpy(index["from"])
        ends = to_numpy(index["to"])
        bounds = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if int(end) > int(start)
        ]
        if bounds:
            return bounds, True

    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes:
        rows = episodes.values() if isinstance(episodes, dict) else episodes
        bounds = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            start = row.get(
                "dataset_from_index",
                row.get("from", row.get("start_index")),
            )
            end = row.get(
                "dataset_to_index",
                row.get("to", row.get("end_index")),
            )
            if start is not None and end is not None and int(end) > int(start):
                bounds.append((int(start), int(end)))
        if bounds:
            return sorted(bounds), True

    return [(0, len(dataset))], False


def select_indices(dataset, bounds, num_frames, val_fraction):
    first_val = int(len(bounds) * (1.0 - val_fraction))
    val_bounds = bounds[first_val:] or [bounds[-1]]
    per_episode = max(1, num_frames // len(val_bounds))

    indices = []
    for start, end in val_bounds:
        low = max(0, int(start) + 1)
        high = min(len(dataset), int(end) - 1)
        if high <= low:
            continue
        take = min(per_episode, high - low)
        indices.extend(
            int(index)
            for index in np.linspace(low, high - 1, take, dtype=np.int64)
        )

    indices = sorted(set(indices))
    if len(indices) > num_frames:
        positions = np.linspace(0, len(indices) - 1, num_frames, dtype=np.int64)
        indices = [indices[index] for index in positions]

    if not indices:
        raise RuntimeError("No evaluation frames selected")

    return indices, val_bounds


def make_infer_repack(data_config):
    inputs = []
    for transform in data_config.repack_transforms.inputs:
        if isinstance(transform, transforms.RepackTransform):
            structure = dict(transform.structure)
            structure.pop("actions", None)
            inputs.append(transforms.RepackTransform(structure=structure))
        else:
            inputs.append(transform)
    return transforms.Group(inputs=inputs)


def evaluate_one(args):
    ckpt_base = Path(os.environ["CKPT_BASE"])
    checkpoint = ckpt_base / str(args.ckpt_step)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(
        data_config,
        cfg.model.action_horizon,
        cfg.model,
    )

    bounds, used_episode_split = episode_bounds(dataset)
    indices, val_bounds = select_indices(
        dataset,
        bounds,
        args.num_frames,
        args.val_fraction,
    )

    print(f"checkpoint={checkpoint}")
    print(f"dataset_len={len(dataset)}")
    print(f"used_episode_split={used_episode_split}")
    print(f"total_episodes={len(bounds)}")
    print(f"eval_episodes={len(val_bounds)}")
    print(f"eval_frames={len(indices)}")

    if not used_episode_split:
        print("WARNING: episode 边界不可用，只能按最后一部分连续帧评估。")
    print("WARNING: 如果这些 episode 参与过训练，本结果不是严格泛化评估。")

    infer_repack = make_infer_repack(data_config)
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint,
        repack_transforms=infer_repack,
    )
    print("POLICY_LOADED", type(policy).__name__)

    records = []
    latencies = []

    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        observation = {
            key: value
            for key, value in sample.items()
            if key not in ("action", "actions", "action_is_pad")
        }

        output = policy.infer(observation)
        prediction = np.asarray(output["actions"], dtype=np.float32)
        ground_truth = np.asarray(sample["action"], dtype=np.float32)

        if prediction.ndim == 1:
            prediction = prediction[None, :]
        if ground_truth.ndim == 1:
            ground_truth = ground_truth[None, :]

        horizon = min(prediction.shape[0], ground_truth.shape[0])
        dimensions = min(prediction.shape[1], ground_truth.shape[1])

        difference = (
            prediction[:horizon, :dimensions]
            - ground_truth[:horizon, :dimensions]
        )
        absolute = np.abs(difference)

        record = {
            "index": int(index),
            "action_mae": float(np.mean(absolute)),
            "action_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "action_max_abs": float(np.max(absolute)),
            "per_dim_mae": np.mean(absolute, axis=0).tolist(),
            "horizon_mae": np.mean(absolute, axis=1).tolist(),
            "finite": bool(np.isfinite(prediction).all()),
        }

        infer_ms = output.get("policy_timing", {}).get("infer_ms")
        if infer_ms is not None:
            record["infer_ms"] = float(infer_ms)
            latencies.append(float(infer_ms))

        records.append(record)

        if count % 10 == 0 or count == len(indices):
            print(
                f"{count}/{len(indices)} "
                f"mae={record['action_mae']:.6f} "
                f"rmse={record['action_rmse']:.6f}"
            )

    steady_latencies = latencies[2:] if len(latencies) > 2 else latencies

    per_dim = np.mean(
        np.asarray([record["per_dim_mae"] for record in records], dtype=np.float32),
        axis=0,
    )
    horizon_length = min(len(record["horizon_mae"]) for record in records)
    horizon_mae = np.mean(
        np.asarray(
            [record["horizon_mae"][:horizon_length] for record in records],
            dtype=np.float32,
        ),
        axis=0,
    )

    result = {
        "config": args.config,
        "checkpoint_step": args.ckpt_step,
        "dataset_len": len(dataset),
        "used_episode_split": used_episode_split,
        "total_episodes": len(bounds),
        "eval_episodes": len(val_bounds),
        "eval_frames": len(records),
        "frame_indices": indices,
        "mean_action_mae": float(np.mean([record["action_mae"] for record in records])),
        "mean_action_rmse": float(np.mean([record["action_rmse"] for record in records])),
        "mean_action_max_abs": float(
            np.mean([record["action_max_abs"] for record in records])
        ),
        "p50_infer_ms": (
            float(np.percentile(steady_latencies, 50))
            if steady_latencies
            else None
        ),
        "p95_infer_ms": (
            float(np.percentile(steady_latencies, 95))
            if steady_latencies
            else None
        ),
        "all_finite": bool(all(record["finite"] for record in records)),
        "per_dim_mae": per_dim.tolist(),
        "horizon_mae": horizon_mae.tolist(),
        "records": records,
    }

    output_path = ckpt_base / "eval" / f"offline_eval_{args.ckpt_step}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("=== RESULT ===")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in (
                    "records",
                    "frame_indices",
                    "per_dim_mae",
                    "horizon_mae",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved={output_path}")


def format_value(value, digits=5):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def compare_results():
    checkpoint_base = Path(os.environ["CKPT_BASE"])
    paths = sorted((checkpoint_base / "eval").glob("offline_eval_*.json"))

    if not paths:
        raise SystemExit("No evaluation JSON files found")

    results = [json.loads(path.read_text()) for path in paths]
    results.sort(key=lambda item: int(item["checkpoint_step"]))

    print()
    print("=" * 100)
    print("CHECKPOINT COMPARISON")
    print("=" * 100)
    print(
        f"{'step':>8} {'frames':>7} {'MAE':>11} {'RMSE':>11} "
        f"{'MaxAbs':>11} {'p95_ms':>10} {'finite':>8} {'best':>6}"
    )
    print("-" * 100)

    best = min(
        results,
        key=lambda item: (
            float(item.get("mean_action_mae", float("inf"))),
            float(item.get("mean_action_rmse", float("inf"))),
        ),
    )
    best_step = int(best["checkpoint_step"])

    for item in results:
        step = int(item["checkpoint_step"])
        print(
            f"{step:>8} "
            f"{int(item.get('eval_frames', 0)):>7} "
            f"{format_value(item.get('mean_action_mae')):>11} "
            f"{format_value(item.get('mean_action_rmse')):>11} "
            f"{format_value(item.get('mean_action_max_abs')):>11} "
            f"{format_value(item.get('p95_infer_ms'), 1):>10} "
            f"{str(bool(item.get('all_finite', False))):>8} "
            f"{'*' if step == best_step else '':>6}"
        )

    dimension_rows = [
        (int(item["checkpoint_step"]), item.get("per_dim_mae") or [])
        for item in results
    ]
    max_dimensions = max((len(values) for _, values in dimension_rows), default=0)

    if max_dimensions:
        print()
        print("=" * 100)
        print("PER-DIMENSION MAE  (* = best for that dimension)")
        print("=" * 100)
        header = f"{'dim':>5} " + " ".join(
            f"{'s' + str(step):>13}" for step, _ in dimension_rows
        )
        print(header)
        print("-" * len(header))

        for dimension in range(max_dimensions):
            values = [
                (
                    float(row[dimension])
                    if dimension < len(row)
                    else float("inf")
                )
                for _, row in dimension_rows
            ]
            best_value = min(values)
            cells = [
                f"{value:>12.5f}{'*' if value == best_value else ' '}"
                for value in values
            ]
            print(f"{dimension:>5} " + " ".join(cells))

    horizon_rows = [
        (int(item["checkpoint_step"]), item.get("horizon_mae") or [])
        for item in results
    ]
    max_horizon = max((len(values) for _, values in horizon_rows), default=0)

    if max_horizon:
        print()
        print("=" * 100)
        print("HORIZON MAE  (10-step bins)")
        print("=" * 100)
        header = f"{'horizon':>10} " + " ".join(
            f"{'s' + str(step):>13}" for step, _ in horizon_rows
        )
        print(header)
        print("-" * len(header))

        for start in range(0, max_horizon, 10):
            end = min(start + 10, max_horizon)
            cells = []
            for _, values in horizon_rows:
                chunk = values[start:end]
                cells.append(sum(chunk) / len(chunk) if chunk else float("inf"))
            print(
                f"{f'{start}-{end - 1}':>10} "
                + " ".join(f"{value:>13.5f}" for value in cells)
            )

    print()
    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    print(f"RECOMMENDED_STEP={best_step}")
    print(f"BEST_MAE={format_value(best.get('mean_action_mae'))}")
    print(f"BEST_RMSE={format_value(best.get('mean_action_rmse'))}")

    if len(results) > 1:
        runner_up = min(
            (item for item in results if item is not best),
            key=lambda item: float(item.get("mean_action_mae", float("inf"))),
        )
        runner_mae = float(runner_up["mean_action_mae"])
        best_mae = float(best["mean_action_mae"])
        if runner_mae > 0:
            improvement = (runner_mae - best_mae) / runner_mae * 100.0
            print(
                f"MAE_IMPROVEMENT_VS_SECOND={improvement:.2f}% "
                f"(step {int(runner_up['checkpoint_step'])})"
            )

    print()
    print(
        "注意：该结果只衡量验证数据上的动作误差，不等于真机任务成功率。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_r1lite")
    parser.add_argument("--ckpt-step", type=int)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_results()
        return

    if args.ckpt_step is None:
        raise SystemExit("Use --ckpt-step or --compare")

    evaluate_one(args)


if __name__ == "__main__":
    main()
