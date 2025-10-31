from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from dinosaw.models import PEModel
from dinosaw.datasets import DatasetTrainStudent, DatasetValStudent


def main():
    train_loader = torch.utils.data.DataLoader(DatasetTrainStudent(), batch_size=32, num_workers=56, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(DatasetValStudent(), batch_size=2)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='trained_models/',
        filename='test-{epoch:02d}-{val_loss:.2f}',
        mode="min"
    )
    
    trainer = Trainer(devices=1, max_epochs=70, gradient_clip_val=1.0, callbacks=checkpoint_callback) #, strategy="ddp"
    trainer.fit(model=PEModel(), train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()