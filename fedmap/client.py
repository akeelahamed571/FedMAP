# fedmap/client.py
import torch
import flwr as fl

class FedMAPClient(fl.client.NumPyClient):
    """
    Extended with:
    ✅ val_loss
    ✅ grad_norm
    ✅ imbalance_ratio
    """

    def __init__(
        self,
        *,
        model,
        train_fn,
        test_fn,
        train_loader,
        val_loader,
        test_loader,
        modality_flags=None,
        local_lr=1e-5,
        local_weight_decay=1e-4,
        local_epochs=5,
        metadata_fn=None,
    ):
        self.model = model
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.modality_flags = modality_flags or {"use_text": 1, "use_image": 1}
        self.local_lr = float(local_lr)
        self.local_weight_decay = float(local_weight_decay)
        self.local_epochs = int(local_epochs)
        self.metadata_fn = metadata_fn

        self.device = next(self.model.parameters()).device

        # Precompute dataset size
        self._n_k = len(self.train_loader.dataset)

        # Precompute class counts
        self._num_pos = None
        self._num_neg = None
        self._compute_class_counts_once()

        # Modality features
        ut = int(self.modality_flags.get("use_text", 1))
        ui = int(self.modality_flags.get("use_image", 1))
        self._m_txt = ut
        self._m_img = ui
        self._modality_div = 1.0 if (ut and ui) else 0.5

    # ---------------------
    # Flower methods
    # ---------------------

    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters, config):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()}, strict=True)

    def fit(self, parameters, config):
        # Load global weights
        self.set_parameters(parameters, config)

        # New optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.local_lr,
            weight_decay=self.local_weight_decay,
        )

        # Save global params for FedProx
        global_params = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}

        # Eval before
        _, acc_before, _ = self.test_fn(self.model, self.val_loader, cfg=self.modality_flags)

        # ---- Track gradient norm ----
        grad_norm = 0.0

        def compute_grad_norm():
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            return total_norm ** 0.5

        # Train locally
        self.train_fn(
            self.model,
            self.train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            cfg=self.modality_flags,
            global_params=global_params,
        )

        grad_norm = compute_grad_norm()

        # Eval after
        val_loss, acc_after, _ = self.test_fn(self.model, self.val_loader, cfg=self.modality_flags)

        # Build metadata
        if self.metadata_fn is not None:
            metrics = self.metadata_fn(
                model=self.model,
                acc_before=acc_before,
                acc_after=acc_after,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                modality_flags=self.modality_flags,
                cached_stats=self._cached_stats(),
            )
        else:
            metrics = self._default_metadata(acc_before=acc_before, acc_after=acc_after)

        # ✅ Add extra metrics
        total = (self._num_pos or 0) + (self._num_neg or 0)
        imbalance_ratio = (
            float(self._num_pos) / total if total > 0 and self._num_pos is not None else 0.0
        )
        metrics.update({
            "val_loss": float(val_loss),
            "grad_norm": float(grad_norm),
            "imbalance_ratio": imbalance_ratio,
        })

        return self.get_parameters(config), self._n_k, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy, metrics = self.test_fn(self.model, self.test_loader, cfg=self.modality_flags)
        return loss, len(self.test_loader.dataset), metrics

    # ---------------------
    # Helpers
    # ---------------------

    def _compute_class_counts_once(self):
        try:
            df = getattr(self.train_loader.dataset, "df", None)
            if df is not None and "label" in df:
                self._num_pos = int((df["label"] == 1).sum())
                self._num_neg = int((df["label"] == 0).sum())
                return
        except Exception:
            pass

        pos = 0
        neg = 0
        try:
            dataset = self.train_loader.dataset
            for i in range(len(dataset)):
                lbl = int(dataset[i]["label"])
                if lbl == 1:
                    pos += 1
                else:
                    neg += 1
        except Exception:
            pos = None
            neg = None

        self._num_pos = pos
        self._num_neg = neg

    def _diversity(self):
        if self._num_pos is None or self._num_neg is None or (self._n_k <= 0):
            return 0.0
        return min(self._num_pos, self._num_neg) / max(1, self._n_k)

    def _cached_stats(self):
        return {
            "n_k": self._n_k,
            "num_pos": self._num_pos,
            "num_neg": self._num_neg,
            "diversity": self._diversity(),
            "m_img": self._m_img,
            "m_txt": self._m_txt,
            "modality_div": self._modality_div,
        }

    def _default_metadata(self, *, acc_before, acc_after):
        stats = self._cached_stats()
        ck = float(acc_after - acc_before)
        pk = float(acc_after)

        return {
            "c_k": ck,
            "p_k": pk,
            "n_k": stats["n_k"],
            "diversity": float(stats["diversity"]),
            "m_img": stats["m_img"],
            "m_txt": stats["m_txt"],
            "modality_div": float(stats["modality_div"]),
            "num_hateful": int(stats["num_pos"]) if stats["num_pos"] is not None else 0,
            "num_nonhateful": int(stats["num_neg"]) if stats["num_neg"] is not None else 0,
        }
