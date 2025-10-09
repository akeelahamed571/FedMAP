import logging
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from flwr.common import parameters_to_ndarrays

import data_preparation
import models
from fedops.server.app import FLServer   # ✅ FedOps wrapper


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Logging/Device
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("server")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("📋 Loaded configuration:\n" + OmegaConf.to_yaml(cfg))

    # Global model
    model = instantiate(cfg.model).to(device)
    model_type = cfg.model_type
    model_name = type(model).__name__

    test_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)
    test_fn = models.test_torch()

    # ✅ global evaluation wrapper
    def global_evaluate(model, test_loader, _cfg):
        loss, acc, metrics = test_fn(model, test_loader, cfg=None)
        logger.info(f"🌍 Global Eval — loss: {loss:.4f}, acc: {acc:.4f}")
        return loss, acc, metrics

    # ✅ Construct FedOps server (no strategy kwarg!)
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=test_loader,
        test_torch=test_fn,   # still pass test function
    )

    logger.info("🔁 Starting FedOps FL server …")
    fl_server.start()


if __name__ == "__main__":
    main()
