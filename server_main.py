import logging
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from flwr.common import parameters_to_ndarrays

import data_preparation
import models
from fedops.server.app import FLServer   # ✅ FedOps server wrapper
from fedmap.strategy import ModalityAwareAggregation  # ✅ your custom strategy


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
    model_type = cfg.model_type
    model_name = type(model).__name__

    test_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)
    test_fn = models.test_torch()

    # ✅ Wrapper for evaluation (loss, acc, metrics)
    def global_evaluate(server_round: int, parameters, _config):
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = dict(
            zip(model.state_dict().keys(), [torch.tensor(v) for v in ndarrays])
        )
        model.load_state_dict(state_dict, strict=True)

        loss, acc, metrics = test_fn(model, test_loader, cfg=None)
        logger.info(f"🌍 [Round {server_round}] Global Eval — loss: {loss:.4f}, acc: {acc:.4f}")
        return float(loss), {"test_accuracy": float(acc), **{k: float(v) for k, v in metrics.items()}}

    # ✅ Build FedOps server with your custom strategy
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=test_loader,
        test_torch=test_fn,
        strategy=ModalityAwareAggregation(
            evaluate_fn=global_evaluate,
            aggregator_path=cfg.server.strategy.aggregator_path,
            input_dim=cfg.server.strategy.input_dim,
            hidden_dim=cfg.server.strategy.hidden_dim,
            aggregator_lr=cfg.server.strategy.aggregator_lr,
            entropy_coeff=cfg.server.strategy.entropy_coeff,
            n_trials_per_round=cfg.server.strategy.n_trials_per_round,
            perf_mix_lambda=cfg.server.strategy.perf_mix_lambda,
            z_clip=cfg.server.strategy.z_clip,
            fraction_fit=cfg.server.strategy.fraction_fit,
            fraction_evaluate=cfg.server.strategy.fraction_evaluate,
            min_fit_clients=cfg.server.strategy.min_fit_clients,
            min_available_clients=cfg.server.strategy.min_available_clients,
            min_evaluate_clients=cfg.server.strategy.min_evaluate_clients,
        )
    )

    logger.info("🔁 Starting FedOps FL server …")
    fl_server.start()


if __name__ == "__main__":
    main()
