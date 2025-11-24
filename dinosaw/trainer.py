from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import torch
import math
from dinosaw.models import PEModel
from dinosaw.datasets import DatasetTrainStudent, DatasetValStudent, GenericDatasetStudent



class AbsPEFader(Callback):
    def __init__(
        self,
        start: float = 1.0,      # Start-Skale
        mid: float = 0.02,       # Ziel am Ende der schnellen Phase
        end: float = 0.0,        # End-Skale
        steps_fast: int = 1000,  # schnelle Phase: Anzahl Optimizer-Schritte
        steps_slow: int = 20000, # langsame Phase: Anzahl Optimizer-Schritte
        slow_kind: str = "cosine",  # "cosine" oder "linear" für Phase 2
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
        self.freeze_pos_embed = freeze_pos_embed
        self.restore_on_fit_end = restore_on_fit_end
        self.vit = None

    def _s(self, step: int) -> float:
        # 2-Phasen-Schedule:
        # Phase 1: linear start -> mid über steps_fast
        # Phase 2: linear oder cosine mid -> end über steps_slow
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
            else:
                raise ValueError(f"Unknown slow_kind:{self.slow_kind}")
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
                # ursprüngliche PE wiederherstellen
                self.vit.pos_embed.copy_(self.vit.pos_embed_base)
            else:
                # Endwert exakt setzen (typisch 0.0)
                self.vit.pos_embed.copy_(self.vit.pos_embed_base * self.end)

def main():
    train_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="train"), batch_size=8, num_workers=32, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="val"), batch_size=32, num_workers=8, pin_memory=True, shuffle=True)

    name = "batch_8_added_alibi_no_wrap"

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
        max_epochs=60, 
        gradient_clip_val=1.0, 
        callbacks=[
            checkpoint_callback, 
            #AbsPEFader(start=1, mid=0.05, end=0, steps_fast=10_000, steps_slow=50_000, freeze_pos_embed=True, restore_on_fit_end=False), 
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
            "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/batch_8-epoch=30-val_loss=0.02.ckpt", 
            use_alibi=True, 
            strict=False
        ),
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader)


if __name__ == "__main__":
    main()