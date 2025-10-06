import logging
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.common import parameters_to_ndarrays

import data_preparation
import models


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Logging/Device
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("server")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("📋 Loaded configuration:\n" + OmegaConf.to_yaml(cfg))

    # Global model for evaluation
    model = instantiate(cfg.model).to(device)
    test_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)
    test_fn = models.test_torch()

    def global_evaluate(server_round: int, parameters, _config):
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = dict(
            zip(model.state_dict().keys(), [torch.tensor(v) for v in ndarrays])
        )
        model.load_state_dict(state_dict, strict=True)

        loss, acc, metrics = test_fn(model, test_loader, cfg=None)
        logger.info(f"🌍 [Round {server_round}] Global Eval — loss: {loss:.4f}, acc: {acc:.4f}")
        return float(loss), {"test_accuracy": float(acc), **{k: float(v) for k, v in metrics.items()}}

    # Strategy from Hydra
    strategy = instantiate(cfg.server.strategy, evaluate_fn=global_evaluate)

    logger.info("🔁 Starting Flower server (0.0.0.0:8080) ...")
    fl.server.start_server(
      server_address="0.0.0.0:8080",
      config=fl.server.ServerConfig(num_rounds=int(cfg.num_rounds)),
      strategy=strategy,
    )


if __name__ == "__main__":
    main()
