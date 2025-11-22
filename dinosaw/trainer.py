from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import torch
from dinosaw.models import PEModel
from dinosaw.datasets import DatasetTrainStudent, DatasetValStudent, GenericDatasetStudent

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='trained_models_in_steps/',
    filename='added_alibi_gradual_fade_abs_pos_embed_long-{epoch:02d}-{val_loss:.2f}',
    mode="min",
)
endcheckpoint_callback = ModelCheckpoint( 
    dirpath='trained_models_in_steps/',
    filename='added_alibi_gradual_fade_abs_pos_embed_long-{epoch:02d}-{val_loss:.2f}_last_epoch',
)

class AbsPEFader(Callback):
    def __init__(self, start=1.0, end=0.0, total_steps=20000, freeze_pos_embed=True, restore_on_fit_end=False):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total_steps = int(total_steps)
        self.freeze_pos_embed = freeze_pos_embed
        self.restore_on_fit_end = restore_on_fit_end
        self.vit = None

    def _s(self, step: int) -> float:
        t = min(1.0, step / max(1, self.total_steps))
        return self.start + t * (self.end - self.start)

    def _find_vit(self, pl_module):
        # Adapt this to your module: PEModel(...).vit.model is your VisionTransformer
        vit = getattr(getattr(pl_module, "vit"), "model", None)
        if vit is None or not hasattr(vit, "pos_embed"):
            # fallback: try to find any module with attribute 'pos_embed'
            for m in pl_module.modules():
                if hasattr(m, "pos_embed"):
                    vit = m
                    break
        if vit is None or not hasattr(vit, "pos_embed"):
            raise RuntimeError("Could not find VisionTransformer with a 'pos_embed' parameter.")
        return vit

    def on_fit_start(self, trainer, pl_module):
        self.vit = self._find_vit(pl_module)
        # Save a pristine copy of PE and optionally freeze the learnable param
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
        # Optional: evaluate with the current schedule value (default)
        # If you want full removal during val once past total_steps, do nothing.
        # If you prefer to always evaluate with s=1, uncomment below:
        # with torch.no_grad():
        #     self.vit.pos_embed.copy_(self.vit.pos_embed_base)
        pass

    def on_fit_end(self, trainer, pl_module):
         with torch.no_grad():
            if self.restore_on_fit_end:
                # restore original PE (not what you want for “retain no-PE”)
                self.vit.pos_embed.copy_(self.vit.pos_embed_base)
            else:
                # force exact end value (usually 0.0) and save it in the checkpoint
                self.vit.pos_embed.copy_(self.vit.pos_embed_base * self.end)

def main():
    train_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="train"), batch_size=128, num_workers=32, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="val"), batch_size=32, num_workers=8, pin_memory=True, shuffle=True)

    
    trainer = Trainer(devices=1, max_epochs=1000, gradient_clip_val=1.0, callbacks=[checkpoint_callback, AbsPEFader(start=1, end=0.0, total_steps=7_000, freeze_pos_embed=True), endcheckpoint_callback], log_every_n_steps=10, accelerator="gpu", val_check_interval=0.2) #, strategy="ddp"
    model = PEModel.load_from_checkpoint("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/nothing-epoch=39-val_loss=0.01.ckpt", remove_pos_embed=False, use_alibi=True, strict=False).to("cuda")
    model_alibi = PEModel.load_from_checkpoint("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/added_alibi-epoch=38-val_loss=0.01.ckpt", remove_pos_embed=False, use_alibi=True, strict=False).to("cuda")
    model_tuned = PEModel.load_from_checkpoint("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps/added_alibi_gradual_fade_abs_pos_embed-epoch=09-val_loss=0.01.ckpt", use_alibi=True, remove_pos_embed=False, strict=False).to("cuda")
    trainer.fit(model=model_alibi, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()