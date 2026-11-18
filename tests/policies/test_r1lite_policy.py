import numpy as np

from openpi.models import model as _model
from openpi.policies import r1lite_policy
from openpi.training import data_loader


def test_r1lite_policy_maps_three_cameras_and_20d_actions() -> None:
    transform = r1lite_policy.R1LiteInputs(model_type=_model.ModelType.PI0)
    data = {
        "observation/image": np.zeros((16, 20, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((12, 16, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.ones((12, 16, 3), dtype=np.uint8),
        "observation/state": np.arange(20, dtype=np.float32),
        "actions": np.zeros((4, 20), dtype=np.float32),
        "prompt": "move the cube",
    }

    result = transform(data)

    assert result["state"].shape == (20,)
    assert result["actions"].shape == (4, 20)
    assert result["prompt"] == "move the cube"
    assert result["image"]["base_0_rgb"].shape == (16, 20, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (12, 16, 3)
    assert result["image_mask"]["base_0_rgb"]
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert result["image_mask"]["right_wrist_0_rgb"]


def test_r1lite_outputs_returns_twenty_action_dimensions() -> None:
    transform = r1lite_policy.R1LiteOutputs()
    model_actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    result = transform({"actions": model_actions})

    assert result["actions"].shape == (2, 20)
    np.testing.assert_array_equal(result["actions"], model_actions[:, :20])


def test_local_dataset_root_can_override_hugging_face_lookup(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENPI_LOCAL_DATASET_ROOT", str(tmp_path))

    assert data_loader._resolve_local_dataset_root() == tmp_path
