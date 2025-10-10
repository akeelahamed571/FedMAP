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
# ❌ MOVE THIS IMPORT DOWN (after we patch Flower)
# from fedops.client.app import FLClientTask

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

    max_len = getattr(cfg, "max_len", 128)
    train_dataset = data_preparation.HatefulMemesDataset(train_df, max_len=max_len)
    val_dataset   = data_preparation.HatefulMemesDataset(val_df,   max_len=max_len)
    test_dataset  = data_preparation.HatefulMemesDataset(test_df,  max_len=max_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
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

    # ---------------- Robust FIX for Flower signal-in-subthread crash ----------------
    try:
        import flwr  # ensure Flower is importable early
        import signal

        patched = False
        try:
            import flwr.client.app as flapp
            if hasattr(flapp, "AppStateTracker"):
                flapp.AppStateTracker.register_signal_handler = lambda *a, **k: None
                patched = True
        except Exception:
            pass
        if not patched:
            try:
                import flwr.client.app as flapp
                if hasattr(flapp, "_app_state_tracker"):
                    flapp._app_state_tracker.register_signal_handler = lambda *a, **k: None
                    patched = True
            except Exception:
                pass
        if not patched:
            try:
                from flwr.client import app_state as _app_state_mod  # type: ignore
                if hasattr(_app_state_mod, "_app_state_tracker"):
                    _app_state_mod._app_state_tracker.register_signal_handler = lambda *a, **k: None
                    patched = True
            except Exception:
                pass
        if not patched:
            _orig_signal = signal.signal
            signal.signal = lambda *a, **k: None
            logger.warning("⚠️ Falling back to global signal.signal no-op to avoid thread signal crash.")
        logger.info("✅ Flower signal handling patched for background execution.")
    except Exception as e:
        logger.warning(f"Could not patch Flower signal handling safely, proceeding anyway: {e}")

    # ---------------- Compatibility shim: patch Flower's start_numpy_client BEFORE importing FedOps ----------------
    try:
        import types, inspect

        # Try both possible locations (depends on Flower version)
        flapp = None
        try:
            import flwr.client.app as flapp  # flwr 1.x
        except Exception:
            flapp = None

        # Fallback: some versions expose start_numpy_client at flwr.client
        if flapp is None or not hasattr(flapp, "start_numpy_client"):
            import flwr.client as flclient
            target_module = flclient
            orig_start = getattr(flclient, "start_numpy_client")
        else:
            target_module = flapp
            orig_start = getattr(flapp, "start_numpy_client")

        def _wrap_get_parameters_if_needed(numpy_client):
            gp = getattr(numpy_client, "get_parameters", None)
            if gp is None:
                return numpy_client
            try:
                sig = inspect.signature(gp)
                if "config" not in sig.parameters:
                    def _gp_with_config(self, config=None, **kwargs):
                        # old-style client: ignore config, call original
                        return gp()
                    numpy_client.get_parameters = types.MethodType(_gp_with_config, numpy_client)
            except Exception:
                def _gp_with_config(self, config=None, **kwargs):
                    return gp()
                numpy_client.get_parameters = types.MethodType(_gp_with_config, numpy_client)
            return numpy_client

        def start_numpy_client_patched(*, server_address: str, client, grpc_max_message_length: int = 536_870_912, **kw):
            client = _wrap_get_parameters_if_needed(client)
            return orig_start(server_address=server_address, client=client,
                              grpc_max_message_length=grpc_max_message_length, **kw)

        # Patch the symbol FedOps will import
        setattr(target_module, "start_numpy_client", start_numpy_client_patched)
        logger.info("✅ Patched Flower start_numpy_client to adapt old NumPyClient.get_parameters signature (pre-FedOps import).")
    except Exception as e:
        logger.warning(f"Could not patch Flower NumPyClient compatibility: {e}")

    # ✅ NOW import FedOps, so it sees the patched start_numpy_client
    from fedops.client.app import FLClientTask  # <-- moved here

    # ---------------- Launch FL client (starts FastAPI; training begins via POST /start) ----------------
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
