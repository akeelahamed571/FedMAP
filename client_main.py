import random
import logging
import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import data_preparation
import models
from fedmap import FedMAPClient

# --- Fix Hydra vs CLI arg conflict ---
import sys, os
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    os.environ["CLIENT_ID"] = sys.argv[1]
    sys.argv = [sys.argv[0]]  # Strip CLI arg so Hydra won’t parse it

# ✅ import FedOps task manager
from fedops.client.app import FLClientTask


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

    # --- Load client-specific partition ---
    client_id = int(os.environ.get("CLIENT_ID", 0))
    logger.info(f"📥 Loading data for client {client_id}")

    train_df, val_df, test_df = data_preparation.load_partition_for_client(client_id)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(data_preparation.HatefulMemesDataset(train_df), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(data_preparation.HatefulMemesDataset(val_df), batch_size=cfg.batch_size)
    test_loader = DataLoader(data_preparation.HatefulMemesDataset(test_df), batch_size=cfg.batch_size)

    # --- Model ---
    model = instantiate(cfg.model).to(device)

    # Train/Test functions
    train_fn = models.train_torch(mu=cfg.fedprox_mu, focal_gamma=cfg.focal_gamma)
    test_fn = models.test_torch()

    # ✅ Build FedMAP Client (your custom logic)
    fedmap_client = FedMAPClient(
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

    # ✅ Prepare FedOps registration (as FedOps expects)
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": type(model).__name__,
        "train_torch": train_fn,
        "test_torch": test_fn,
    }

    # ✅ Create FedOps client task
    fl_client = FLClientTask(cfg, registration)

    # 🔀 Monkey-patch FedMAP client into FedOps task
    fl_client.client = fedmap_client.to_client()

    logger.info(f"🔁 [Client {client_id}] Starting FedOps FL client …")
    fl_client.start()


if __name__ == "__main__":
    main()
