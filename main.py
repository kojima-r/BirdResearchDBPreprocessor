import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import itertools

from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import BirdSongDataset
from model import Transfer_Cnn14



class ClassificationTask(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.val_losses=[]
        self.train_losses=[]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean().item()
        tensorboard_logs = {'val_loss': avg_loss}
        self.val_losses.append(avg_loss)
        print("===")
        print("avg_loss:",avg_loss)
        print("avg_acc:",avg_acc)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.train_losses.append(avg_loss)
        return None
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        lr=self.args.learning_rate
        if self.args.optimizer=="adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif self.args.optimizer=="sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            pass
            
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClassificationTask")
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        return parent_parser


from argparse import ArgumentParser
num_workers=4
sample_rate=32000
window_size=1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
def get_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ClassificationTask.add_model_specific_args(parser)
    subparser = parser.add_argument_group("Other")
    subparser.add_argument("--batch_size", type=int, default=32)
    subparser.add_argument("--valid_rate", type=float, default=0.8)
    subparser.add_argument("--freeze_base", action="store_true", default=False)
    #subparser.add_argument("--pretrain", action="store_true", default=False)
    args = parser.parse_args()
    return args


def pred():
    args = get_args()
    dataset = BirdSongDataset()
    n_samples = len(dataset)
    train_size = int(len(dataset) * args.valid_rate)
    val_size = n_samples - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print("#all:",len(dataset))
    print("#train:",len(train_dataset))
    print("#valid:",len(valid_dataset))
    print("#labels:",len(dataset.label_mapping))
    
    classes_num=len(dataset.label_mapping)
    batch_size=args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers)

    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    #model = Transfer_Cnn14.load_from_checkpoint("best_models/sample-epoch=121-val_loss=0.44.ckpt")
    ###
    task=ClassificationTask.load_from_checkpoint(checkpoint_path="best_models/sample-epoch=121-val_loss=0.44.ckpt",model=model,args=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(task, dataloaders=[valid_loader])
    print(trainer.model.model)
    #print(trainer.model.model.state_dict())
    model_path="best_models/best_model.pth"
    torch.save(trainer.model.model.to('cpu').state_dict(),model_path )

    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    for x,_ in valid_loader:
        with torch.no_grad():   
            pred_y=model(x)
            print(pred_y)
            print(pred_y.shape)
            break


def train():
    args = get_args()
    dataset = BirdSongDataset()
    n_samples = len(dataset)
    train_size = int(len(dataset) * args.valid_rate)
    val_size = n_samples - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print("#all:",len(dataset))
    print("#train:",len(train_dataset))
    print("#valid:",len(valid_dataset))
    print("#labels:",len(dataset.label_mapping))
    
    classes_num=len(dataset.label_mapping)
    batch_size=args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers)

    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    model.load_from_pretrain("Cnn14_mAP=0.431.pth")
    task=ClassificationTask(model,args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="best_models/",
        filename="sample-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        )

    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="best_models/",
        filename="best_model",
        save_top_k=1,
        mode="min",
        )


    #trainer = pl.Trainer(gpus=1,log_every_n_steps=1,max_epochs=10,enable_progress_bar=False)
    trainer = pl.Trainer.from_argparse_args(args,callbacks=[checkpoint_callback,best_checkpoint_callback])
    trainer.fit(task, train_dataloaders=train_loader,val_dataloaders=valid_loader)

    ### loss plot
    print(task.train_losses)
    print(task.val_losses)
    with open("loss_log.tsv","w") as ofp:
        for l1,l2 in itertools.zip_longest(task.train_losses,task.val_losses,fillvalue=''):
            ofp.write(str(l1)+"\t"+str(l2)+"\n")

    ### save best model
    task=ClassificationTask.load_from_checkpoint(checkpoint_path="best_models/best_model.ckpt",model=model,args=args)
    trainer = pl.Trainer.from_argparse_args(args)
    model_path="best_models/best_model.pth"
    torch.save(trainer.model.model.to('cpu').state_dict(),model_path )

if __name__ == "__main__":
    #train()
    pred()

