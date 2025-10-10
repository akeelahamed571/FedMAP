# client_main.py
import sys
import random
import hydra
import numpy as np
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import data_preparation
import models
from fedops.client import client_utils
from fedops.client.app import FLClientTask


# --- Extract client_id BEFORE Hydra parses ---
extra_args = []
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    extra_args.append(int(sys.argv[1]))
    sys.argv = [sys.argv[0]] + sys.argv[2:]


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ---------------- Logging ----------------
    handlers_list = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)8.8s] %(message)s",
        handlers=handlers_list,
    )
    logger = logging.getLogger(__name__)

    # ---------------- Reproducibility ----------------
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print(OmegaConf.to_yaml(cfg))
    client_id = extra_args[0] if extra_args else 0
    logger.info(f"🚀 Starting client {client_id}")

    # ---------------- Data ----------------
    train_df, val_df, test_df = data_preparation.load_partition_for_client(client_id)

    train_dataset = data_preparation.HatefulMemesDataset(train_df, max_len=getattr(cfg, "max_len", 128))
    val_dataset   = data_preparation.HatefulMemesDataset(val_df,   max_len=getattr(cfg, "max_len", 128))
    test_dataset  = data_preparation.HatefulMemesDataset(test_df,  max_len=getattr(cfg, "max_len", 128))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    logger.info("✅ Data loaded")

    # ---------------- Model ----------------
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    train_torch = models.train_torch(mu=cfg.get("fedprox_mu", 0.0), focal_gamma=cfg.get("focal_gamma", 0.0))
    test_torch  = models.test_torch()

    # ---------------- Local checkpoint restore ----------------
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        logger.info("⬇️ Loading latest local model …")
        model = client_utils.download_local_model(
            model_type=model_type, task_id=task_id, listdir=local_list, model=model
        )

    # ---------------- Registration dict ----------------
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch,
    }

    # ---------------- Launch FL client ----------------
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
