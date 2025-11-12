from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from dinosaw.models import PEModel
from dinosaw.datasets import DatasetTrainStudent, DatasetValStudent, GenericDatasetStudent


def main():
    train_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="train"), batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(GenericDatasetStudent(base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224", split="val"), batch_size=32, num_workers=8, pin_memory=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='trained_models/',
        filename='test-{epoch:02d}-{val_loss:.2f}',
        mode="min"
    )
    
    trainer = Trainer(devices=1, max_epochs=20, gradient_clip_val=1.0, callbacks=checkpoint_callback) #, strategy="ddp"
    trainer.fit(model=PEModel(), train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()