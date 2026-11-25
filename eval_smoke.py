import os
import numpy as np

from openpi import transforms
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader

CKPT = os.environ["CKPT"]
cfg = _config.get_config("pi0_r1lite")
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

dataset = data_loader.create_torch_dataset(
    data_config,
    cfg.model.action_horizon,
    cfg.model,
)
sample = dataset[0]

print("sample_keys:")
for key in sorted(sample):
    value = sample[key]
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    print(f"  {key}: shape={shape}")

# 训练 repack 含 actions；推理输入没有真实 action，必须去掉。
inputs = []
for transform in data_config.repack_transforms.inputs:
    if isinstance(transform, transforms.RepackTransform):
        structure = dict(transform.structure)
        structure.pop("actions", None)
        inputs.append(transforms.RepackTransform(structure=structure))
    else:
        inputs.append(transform)

repack = transforms.Group(inputs=inputs)

policy = policy_config.create_trained_policy(
    cfg,
    CKPT,
    repack_transforms=repack,
)

observation = {
    key: value
    for key, value in sample.items()
    if key not in ("action", "actions", "action_is_pad")
}

output = policy.infer(observation)
pred = np.asarray(output["actions"], dtype=np.float32)
gt = np.asarray(sample["action"], dtype=np.float32)

print("POLICY_LOADED", type(policy).__name__)
print("prediction_shape", pred.shape)
print("ground_truth_shape", gt.shape)
print("prediction_finite", bool(np.isfinite(pred).all()))

horizon = min(pred.shape[0], gt.shape[0])
dims = min(pred.shape[1], gt.shape[1])
difference = pred[:horizon, :dims] - gt[:horizon, :dims]

print("action_mae", float(np.mean(np.abs(difference))))
print("action_rmse", float(np.sqrt(np.mean(difference ** 2))))
print("SAMPLE_INFER_OK")
