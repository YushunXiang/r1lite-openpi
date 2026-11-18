import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 20


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class R1LiteInputs(transforms.DataTransformFn):
    """Map R1 Lite LeRobot rows to OpenPI's three-camera observation format."""

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(
                    data["observation/left_wrist_image"]
                ),
                "right_wrist_0_rgb": _parse_image(
                    data["observation/right_wrist_image"]
                ),
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(True),
                "right_wrist_0_rgb": np.bool_(True),
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class R1LiteOutputs(transforms.DataTransformFn):
    """Strip OpenPI action padding when producing R1 Lite robot actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}
