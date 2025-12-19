from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from sorting_transformers.data.collate import collate_batch
from sorting_transformers.data.sorting_dataset import build_dataset_from_config
from sorting_transformers.models.variants import build_model
from sorting_transformers.train.evaluate import evaluate_model
from sorting_transformers.utils.io import append_csv_row, ensure_dir, save_json, save_metadata
from sorting_transformers.utils.seed import resolve_device, select_amp


class Trainer:
    def __init__(self, cfg: Dict, run_dir: Path, resume_ckpt: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.resume_ckpt = resume_ckpt

        self.device = resolve_device(cfg["device"])
        self.model = build_model(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg["pad_token_id"])
        self.scaler = torch.cuda.amp.GradScaler(enabled=select_amp(self.device, cfg["train"]["amp"]))

        self.global_step = 0
        self.tokens_seen = 0
        self.best_metric = -math.inf
        self.best_step = 0

        if resume_ckpt:
            self._load_checkpoint(resume_ckpt)

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        self.best_metric = checkpoint.get("best_metric", -math.inf)
        self.best_step = checkpoint.get("best_step", 0)

    def _make_loader(self, split: str, seed_offset: int) -> DataLoader:
        dataset = build_dataset_from_config(self.cfg, split=split, seed_offset=seed_offset)
        generator = torch.Generator().manual_seed(self.cfg["seed"] + seed_offset)
        return DataLoader(
            dataset,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
            collate_fn=lambda batch: collate_batch(batch, self.cfg["pad_token_id"]),
        )

    def _save_checkpoint(self, name: str, val_metric: float) -> None:
        ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        ckpt_path = ckpt_dir / name
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "config": self.cfg,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "val_metric": val_metric,
        }
        torch.save(checkpoint, ckpt_path)

    def train(self) -> Dict[str, float]:
        train_loader = self._make_loader("train", seed_offset=0)
        val_loader = self._make_loader("val", seed_offset=10_000)

        logs_dir = ensure_dir(self.run_dir / "logs")
        train_log_path = logs_dir / "train.csv"
        val_log_path = logs_dir / "val.csv"

        save_csv_flag = self.cfg.get("logging", {}).get("save_csv", True)
        save_json_flag = self.cfg.get("logging", {}).get("save_json", True)

        if save_json_flag:
            save_metadata(self.run_dir / "metadata.json", self.cfg)

        max_steps = self.cfg["train"].get("max_steps")
        epochs = self.cfg["train"].get("epochs")
        tokens_budget = self.cfg["train"].get("tokens_budget")

        def should_stop() -> bool:
            if max_steps is not None and self.global_step >= max_steps:
                return True
            if tokens_budget is not None and self.tokens_seen >= tokens_budget:
                return True
            return False

        def train_iterator():
            while True:
                for batch in train_loader:
                    yield batch

        if epochs:
            epoch_iter = range(epochs)
            data_iter = None
        else:
            epoch_iter = range(1)
            data_iter = train_iterator()

        for _ in epoch_iter:
            if epochs:
                data_iter = iter(train_loader)
            for batch in data_iter:
                if should_stop():
                    break

                self.model.train()
                input_ids = batch["input_ids"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                decoder_target = batch["decoder_target"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

                src_key_padding_mask = ~attention_mask
                tgt_key_padding_mask = ~decoder_attention_mask

                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    logits = self.model(
                        input_ids,
                        decoder_input,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                    )
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        decoder_target.reshape(-1),
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.cfg["train"]["grad_clip"] is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg["train"]["grad_clip"],
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                tokens = decoder_target.ne(self.cfg["pad_token_id"]).sum().item()
                self.tokens_seen += tokens
                self.global_step += 1

                if save_csv_flag:
                    append_csv_row(
                        train_log_path,
                        fieldnames=["step", "loss", "tokens", "lr"],
                        row={
                            "step": self.global_step,
                            "loss": loss.item(),
                            "tokens": tokens,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        },
                    )

                eval_every = self.cfg["train"].get("eval_every")
                if eval_every and self.global_step % eval_every == 0:
                    val_metrics = evaluate_model(
                        self.model,
                        val_loader,
                        pad_token_id=self.cfg["pad_token_id"],
                        bos_token_id=self.cfg["bos_token_id"],
                        eos_token_id=self.cfg["eos_token_id"],
                        device=self.device,
                    )
                    val_metrics["step"] = self.global_step
                    if save_csv_flag:
                        append_csv_row(
                            val_log_path,
                            fieldnames=["step"] + [key for key in val_metrics if key != "step"],
                            row=val_metrics,
                        )

                    metric_value = val_metrics.get("exact_match", 0.0)
                    if metric_value > self.best_metric:
                        self.best_metric = metric_value
                        self.best_step = self.global_step
                        self._save_checkpoint("best.ckpt", metric_value)

                ckpt_every = self.cfg["train"].get("ckpt_every")
                if ckpt_every and self.global_step % ckpt_every == 0:
                    self._save_checkpoint("last.ckpt", self.best_metric)

            if should_stop():
                break

        self._save_checkpoint("last.ckpt", self.best_metric)
        summary = {
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
        }
        if save_json_flag:
            save_json(summary, self.run_dir / "summary.json")
        return summary
