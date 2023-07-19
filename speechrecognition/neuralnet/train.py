import os
import ast
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from model import SpeechRecognition
from dataset import Data, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class SpeechModule(LightningModule):

    def __init__(self, model, learning_rate=1e-3):
        super(SpeechModule, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.learning_rate = learning_rate

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.50, patience=6)
        return [optimizer], [scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizers().param_groups[0]['lr']}
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        train_dataset = Data(json_path=self.args.train_file, **d_params)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        test_dataset = Data(json_path=self.args.valid_file, **d_params, valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          collate_fn=collate_fn_padd,
                          pin_memory=True)


def checkpoint_callback(save_model_path):
    return ModelCheckpoint(
        dirpath=save_model_path,
        filename='',
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )


def main(train_file, valid_file, save_model_path, load_model_from=None, resume_from_checkpoint=None,
         logdir='tb_logs', epochs=10, batch_size=64, learning_rate=1e-3, valid_every=1000,
         hparams_override={}, dparams_override={}, gpus=1):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(hparams_override)
    model = SpeechRecognition(**h_params)

    if load_model_from:
        speech_module = SpeechModule.load_from_checkpoint(load_model_from, model=model, learning_rate=learning_rate)
    else:
        speech_module = SpeechModule(model, learning_rate=learning_rate)

    logger = TensorBoardLogger(logdir, name='speech_recognition')

    checkpoint_callback_obj = checkpoint_callback(save_model_path)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cuda',  # Use 'ddp' for distributed training
        num_nodes=1,
        logger=logger,
        gradient_clip_val=1.0,
        val_check_interval=valid_every,
        callbacks=[checkpoint_callback_obj]
        #resume_from_checkpoint=resume_from_checkpoint
    )
    if resume_from_checkpoint:
        trainer.resume_from_checkpoint(resume_from_checkpoint)

    trainer.fit(speech_module)

    trainer.fit(speech_module)


# Usage:
train_file = "train.json"
valid_file = "test.json"
save_model_path = "model.pth"

main(train_file=train_file, valid_file=valid_file, save_model_path=save_model_path)




