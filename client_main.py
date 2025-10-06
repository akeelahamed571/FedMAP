import random
import logging
import numpy as np
import torch
import flwr as fl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import data_preparation
import models
from fedmap import FedMAPClient


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Logging & Seeds
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("client")
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    logger.info("📋 Loaded configuration:\n" + OmegaConf.to_yaml(cfg))

    # Data
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        batch_size=cfg.batch_size
    )

    # Model
    model = instantiate(cfg.model).to(device)

    # Train/Test functions
    train_fn = models.train_torch(mu=cfg.fedprox_mu, focal_gamma=cfg.focal_gamma)
    test_fn = models.test_torch()

    # FedMAP Client
    client = FedMAPClient(
        model=model,
        train_fn=train_fn,
        test_fn=test_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        modality_flags={"use_text": 1, "use_image": 1},
        local_lr=float(cfg.lr),
        local_weight_decay=float(cfg.weight_decay),
        local_epochs=int(cfg.num_epochs),
        metadata_fn=None,
    )

    logger.info("🔁 Connecting to Flower server at localhost:8080 ...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
    main()
