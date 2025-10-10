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
from fedmap.client import FedMAPClient


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ✅ Catch CLI positional arg like `python client_main.py 0`
    if len(sys.argv) > 1:
        try:
            cfg.client_id = int(sys.argv[1])
        except ValueError:
            raise ValueError(f"Invalid client_id argument: {sys.argv[1]}")

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

    # ---------------- Data Loading ----------------
    client_id = cfg.client_id
    train_df, val_df, test_df = data_preparation.load_partition_for_client(client_id)

    train_dataset = data_preparation.HatefulMemesDataset(train_df, max_len=cfg.max_len)
    val_dataset   = data_preparation.HatefulMemesDataset(val_df,   max_len=cfg.max_len)
    test_dataset  = data_preparation.HatefulMemesDataset(test_df,  max_len=cfg.max_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    logger.info(f"✅ Client {client_id} data loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # ---------------- Model ----------------
    model = instantiate(cfg.model)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"✅ Initialized model: {type(model).__name__}")

    # ---------------- Train & Test Functions ----------------
    train_torch = models.train_torch(mu=cfg.get("fedprox_mu", 0.0))
    test_torch  = models.test_torch()

    # ---------------- Flower Client ----------------
    fl_client = FedMAPClient(
        model=model,
        train_fn=train_torch,
        test_fn=test_torch,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        modality_flags={"use_text": 1, "use_image": 1},
        local_lr=cfg.get("local_lr", 1e-5),
        local_weight_decay=cfg.get("local_weight_decay", 1e-4),
        local_epochs=cfg.get("local_epochs", 1),
    )

    import flwr as fl
    fl.client.start_client(
        server_address=cfg.server_address,
        client=fl_client
    )


if __name__ == "__main__":
    main()
