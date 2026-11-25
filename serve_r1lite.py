import argparse
import logging
from pathlib import Path

from openpi import transforms
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config

DEFAULT_CHECKPOINT = (
    "/inspire/hdd/project/robot-reasoning/czxs253130583/"
    "openpi_r1lite_checkpoints/pi0_r1lite/"
    "r1lite-full-lora-20261119/29999"
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_r1lite")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    infer_repack = make_infer_repack(data_config)

    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint,
        repack_transforms=infer_repack,
    )

    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("checkpoint=%s", checkpoint)
    logging.info("infer_repack=%s", infer_repack)
    logging.info("SERVER_READY host=%s port=%s", args.host, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
