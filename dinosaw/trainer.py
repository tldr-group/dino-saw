from lightning import Trainer
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import torch
from typing import Callable, Optional
import torch.nn as nn
import math
from dinosaw.models import PEModel
from dinosaw.datasets import DatasetTrainStudent, DatasetValStudent, GenericDatasetStudent



class AbsPEFader(Callback):
    def __init__(
        self,
        start: float = 1.0,       # Start-Skale
        mid: float = 0.02,        # Ziel am Ende der schnellen Phase
        end: float = 0.0,         # End-Skale
        steps_fast: int = 1000,   # schnelle Phase: Anzahl Optimizer-Schritte
        steps_slow: int = 20000,  # langsame Phase: Anzahl Optimizer-Schritte
        slow_kind: str = "cosine",  # "cosine", "linear" oder "log"
        log_k: float = 9.0,       # Form-Parameter für logarithmischen Abfall (größer = schneller früher Abfall)
        freeze_pos_embed: bool = True,
        restore_on_fit_end: bool = False
    ):
        super().__init__()
        self.start = float(start)
        self.mid = float(mid)
        self.end = float(end)
        self.steps_fast = max(1, int(steps_fast))
        self.steps_slow = max(1, int(steps_slow))
        self.slow_kind = slow_kind
        self.log_k = float(log_k)
        self.freeze_pos_embed = freeze_pos_embed
        self.restore_on_fit_end = restore_on_fit_end
        self.vit = None

    def _s(self, step: int) -> float:
        # 2-Phasen-Schedule:
        # Phase 1: linear start -> mid über steps_fast
        # Phase 2: linear/cosine/logarithmisch mid -> end über steps_slow
        if step <= self.steps_fast:
            t = step / self.steps_fast
            p = self.start + t * (self.mid - self.start)
        else:
            t = min(1.0, (step - self.steps_fast) / self.steps_slow)
            if self.slow_kind == "linear":
                p = self.mid + t * (self.end - self.mid)
            elif self.slow_kind == "cosine":
                # cosine-anneal: 1 -> 0 auf [0,1]
                cos_t = 0.5 * (1 + math.cos(math.pi * t))
                p = self.end + (self.mid - self.end) * cos_t
            elif self.slow_kind in ("log", "logarithmic", "logarithmisch"):
                # Normalisierte Log-Kurve z in [0,1]
                # z(0)=0, z(1)=1; größere log_k -> stärkerer früher Abfall
                if self.mid == self.end:
                    p = self.end
                else:
                    k = max(1e-6, self.log_k)
                    z = math.log1p(k * t) / math.log1p(k)   # 0..1, konkav
                    p = self.mid + (self.end - self.mid) * z
            else:
                raise ValueError(f"Unknown slow_kind")
        return float(min(1.0, max(0.0, p)))

    def _find_vit(self, pl_module):
        # Passe das an deine Struktur an: z. B. pl_module.vit.model
        vit = getattr(getattr(pl_module, "vit", None), "model", None)
        if vit is None or not hasattr(vit, "pos_embed"):
            for m in pl_module.modules():
                if hasattr(m, "pos_embed"):
                    vit = m
                    break
        if vit is None or not hasattr(vit, "pos_embed"):
            raise RuntimeError("Could not find VisionTransformer with a 'pos_embed' parameter.")
        return vit

    def on_fit_start(self, trainer, pl_module):
        self.vit = self._find_vit(pl_module)
        # Ursprüngliche PE sichern (wird nicht mitgespeichert)
        if not hasattr(self.vit, "pos_embed_base"):
            self.vit.register_buffer("pos_embed_base", self.vit.pos_embed.detach().clone(), persistent=False)
        if self.freeze_pos_embed:
            self.vit.pos_embed.requires_grad_(False)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        s = self._s(trainer.global_step)
        with torch.no_grad():
            self.vit.pos_embed.copy_(self.vit.pos_embed_base * s)
        pl_module.log("abs_pe_scale", s, prog_bar=True, logger=True)

    def on_validation_start(self, trainer, pl_module):
        # Standard: mit aktuellem s validieren.
        # Wenn du bei Val immer s=1 willst, hier einkommentieren:
        # with torch.no_grad():
        #     self.vit.pos_embed.copy_(self.vit.pos_embed_base)
        pass

    def on_fit_end(self, trainer, pl_module):
        with torch.no_grad():
            if self.restore_on_fit_end:
                self.vit.pos_embed.copy_(self.vit.pos_embed_base)
            else:
                self.vit.pos_embed.copy_(self.vit.pos_embed_base * self.end)


class PositionalDropoutScheduler(Callback):
    def __init__(
        self,
        target_attr: str = "model.pos_drop",  # dot-path to your dropout (e.g., "model.pos_drop" or just "pos_drop")
        p_start: float = 0.0,
        p_end: float = 0.1,
        schedule: str | Callable[[int, int, float, float], float] = "linear",
        max_steps: Optional[int] = None,
        clamp: tuple[float, float] = (0.0, 1.0),
        log_name: str = "pos_drop_p",
    ):
        self.target_attr = target_attr
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.schedule = schedule
        self.max_steps = max_steps
        self.clamp = clamp
        self.log_name = log_name
        self._drop_module: Optional[nn.Module] = None
        self.vit=None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: Optional[str] = None):
        # Resolve the dropout module
        self._drop_module = self._resolve_target(pl_module, self.target_attr)
        if self._drop_module is None or not hasattr(self._drop_module, "p"):
            raise RuntimeError(f"Could not find a Dropout with attribute 'p' at'")

        

        # Determine total steps once we know the trainer
        if self.max_steps is None:
            # PL provides an estimate of total optimization steps
            self.max_steps = trainer.estimated_stepping_batches or trainer.max_steps

        # Initialize p
        self._drop_module.p = float(self.p_start)

    def _find_vit(self, pl_module):
        # Passe das an deine Struktur an: z. B. pl_module.vit.model
        vit = getattr(getattr(pl_module, "vit", None), "model", None)
        if vit is None or not hasattr(vit, "pos_embed"):
            for m in pl_module.modules():
                if hasattr(m, "pos_embed"):
                    vit = m
                    break
        if vit is None or not hasattr(vit, "pos_embed"):
            raise RuntimeError("Could not find VisionTransformer with a 'pos_embed' parameter.")
        return vit

    def on_fit_start(self, trainer, pl_module):
        self.vit = self._find_vit(pl_module)

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int):
        step = trainer.global_step  # 0-based
        new_p = self._compute_p(step, self.max_steps)
        if new_p == 1:
            with torch.no_grad():
                self.vit.pos_embed = torch.nn.Parameter(torch.zeros_like(self.vit.pos_embed))
        #self._drop_module.p = new_p
        self.vit.pos_drop = torch.nn.modules.Dropout(p=new_p)
        # Optional logging
        pl_module.log(self.log_name, new_p, on_step=True, prog_bar=True, logger=True)

    # --- helpers ---
    def _resolve_target(self, root: nn.Module, path: str) -> Optional[nn.Module]:
        # Try dot-path first
        cur = root
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, nn.Module):
                return cur
        except AttributeError:
            pass
        # Fallback: search by suffix name
        for name, m in root.named_modules():
            if name.endswith(path) and isinstance(m, nn.Module):
                return m
        return None

    def _compute_p(self, step: int, max_steps: int) -> float:
        # Normalize step to [0, 1]
        denom = max(1, (max_steps or 1) - 1)
        t = min(1.0, step / denom)

        if callable(self.schedule):
            p = self.schedule(step, max_steps, self.p_start, self.p_end)
        elif self.schedule == "linear":
            p = self.p_start + (self.p_end - self.p_start) * t
        elif self.schedule == "cosine":
            p = self.p_end - (self.p_end - self.p_start) * 0.5 * (1.0 + math.cos(math.pi * t))
        elif self.schedule == "step":
            total_bins = 4
            bins = int(math.floor(t * total_bins))  # 0..10
            p = self.p_start + 0.25 * bins

            # Respect p_end as a cap (if p_end > p_start it's an upper cap; if p_end < p_start it's a lower cap)
            if self.p_end >= self.p_start:
                p = min(p, self.p_end)
            else:
                p = max(p, self.p_end)
        else:
            raise ValueError(f"Unknown schedule:")

        lo, hi = self.clamp
        return float(max(lo, min(hi, p)))

def main():
    train_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="train"), batch_size=8, num_workers=32, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="val"), batch_size=32, num_workers=8, pin_memory=True, shuffle=True)

    name = "Adam_batch_8_mse_after_train_Norm_lr_e-5"

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='trained_models_in_steps/',
        filename=name + '-{epoch:02d}-{val_loss:.2f}',
        mode="min",
    )
    endcheckpoint_callback = ModelCheckpoint( 
        dirpath='trained_models_in_steps/',
        filename=name + '-{epoch:02d}-{val_loss:.2f}_last_epoch',
    )
    
    trainer = Trainer(
        devices=1, 
        max_epochs=40, 
        gradient_clip_val=1.0, 
        callbacks=[
            checkpoint_callback, 
            #PositionalDropoutScheduler(target_attr="model.pos_drop", p_start=0, p_end=1, schedule="step", max_steps=10_000), 
            endcheckpoint_callback
        ], 
        log_every_n_steps=20, 
        accelerator="gpu", 
        val_check_interval=0.2
    ) #, strategy="ddp" #AbsPEFader(start=1, end=0.0, total_steps=7_000, freeze_pos_embed=True),

    # model = PEModel.load_from_checkpoint("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/nothing-epoch=39-val_loss=0.01.ckpt", remove_pos_embed=False, use_alibi=True, strict=False).to("cuda")
    # model_alibi = PEModel.load_from_checkpoint("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/added_alibi-epoch=38-val_loss=0.01.ckpt", remove_pos_embed=False, use_alibi=True, strict=False).to("cuda")
    # model_tuned = PEModel.load_from_checkpoint("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/added_alibi_gradual_fade_abs_pos_embed-epoch=09-val_loss=0.01.ckpt", use_alibi=True, remove_pos_embed=False, strict=False).to("cuda")

        
    trainer.fit(
        model=PEModel.load_from_checkpoint(
            "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/ONLY_NORM_Adam_batch_8_mse_with_0.25_step_dropout_120k_steps_lr_e-5-epoch=15-val_loss=1.42_last_epoch.ckpt", 
            remove_pos_embed=True,
            use_alibi=True, 
            strict=False,
            loss_func="mse"
        ),
        #model = PEModel(loss_func="cosine_embedding"),
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader)


if __name__ == "__main__":
    main()