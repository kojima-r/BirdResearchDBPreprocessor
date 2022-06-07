import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torch.nn.functional as F
import itertools

from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import BirdSongDataset
from model import Transfer_Cnn14
from class_balanced_loss import CB_loss


from argparse import ArgumentParser

num_workers=4
sample_rate=16000
window_size=512
hop_size=160
mel_bins=64
fmin=50
fmax=8000
pretrain_path="Cnn14_16k_mAP=0.438.pth"

"""
# 32k
sample_rate=32000
window_size=1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
pretrain_path="Cnn14_mAP=0.431.pth"
"""



class ClassificationTask(pl.LightningModule):
    def __init__(self, model, args,  samples_per_cls=None, no_of_classes=None):
        super().__init__()
        self.model = model
        self.args = args
        self.val_losses=[]
        self.train_losses=[]
        self.samples_per_cls=samples_per_cls
        self.no_of_classes=no_of_classes
        self.loss_type='focal'
        self.loss_beta=0.9999
        self.loss_gamma=0.0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        ## loss function
        loss = CB_loss(y, y_hat, self.samples_per_cls, self.no_of_classes, self.loss_type, self.loss_beta, self.loss_gamma, device="cuda")
        #loss = F.cross_entropy(y_hat, y)
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
        print("avg_val_loss:",avg_loss)
        print("avg_val_acc:",avg_acc)
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
        #loss = F.cross_entropy(y_hat, y)
        loss = CB_loss(y, y_hat, self.samples_per_cls, self.no_of_classes, self.loss_type, self.loss_beta, self.loss_gamma, device="cuda")
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

def get_args():
    parser = ArgumentParser()
    parser.add_argument("mode",  choices=['train', 'pred', 'test', 'train_cv'])
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ClassificationTask.add_model_specific_args(parser)
    subparser = parser.add_argument_group("Other")
    subparser.add_argument("--batch_size", type=int, default=32)
    subparser.add_argument("--valid_rate", type=float, default=0.8)
    subparser.add_argument("--freeze_base", action="store_true", default=False)
    #subparser.add_argument("--pretrain", action="store_true", default=False)
    args = parser.parse_args()
    return args

def count_n_per_label(dataset,classes_num, pseudo_count=1):
    counter=np.zeros((classes_num,))+pseudo_count
    for _,v in dataset:
        counter[v.item()]+=1
    return counter


def pred(args):
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
    ckpt_path="best_models/best_model.ckpt"
    task=ClassificationTask.load_from_checkpoint(checkpoint_path=ckpt_path,model=model,args=args)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(task, dataloaders=[valid_loader])
    print(trainer.model.model)
    #print(trainer.model.model.state_dict())
    model_path="best_models/best_model.pth"
    torch.save(trainer.model.model.to('cpu').state_dict(),model_path )

    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    pred_y_all=[]
    y_all=[]
    for x,y in valid_loader:
        with torch.no_grad():   
            pred_y_prob = model(x)
            #print(pred_y_prob)
            #print(pred_y_prob.shape)
            pred_y = np.argmax(pred_y_prob,axis=1)
            pred_y_all.extend(pred_y)
        y_all.extend(y)
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_all, pred_y_all)
    print("accuracy:",acc)

def train(args):
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
    model.load_from_pretrain(pretrain_path)
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
    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    task=ClassificationTask.load_from_checkpoint(checkpoint_path="best_models/best_model.ckpt",model=model,args=args)
    trainer = pl.Trainer.from_argparse_args(args)
    model_path="best_models/best_model.pth"
    torch.save(trainer.model.model.to('cpu').state_dict(),model_path )

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def train_cv(args):
    dataset = BirdSongDataset()
    n_samples = len(dataset)
    kf = KFold(n_splits=5,shuffle=True)
    classes_num=len(dataset.label_mapping)
    batch_size=args.batch_size
    dummyX = list(range(n_samples))
    result=[]
    for fold, (train_valid_index, test_index) in enumerate(kf.split(dummyX)):
        train_valid_dataset = Subset(dataset, train_valid_index)
        test_dataset = Subset(dataset, test_index)
        train_size = int(len(train_valid_index) * args.valid_rate)
        val_size = len(train_valid_index)- train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [train_size, val_size])
        n_per_label = count_n_per_label(train_valid_dataset, classes_num)

        print("## fold {}".format(fold))
        print("#all:",    len(dataset))
        print("#train:",  len(train_dataset))
        print("#valid:",  len(valid_dataset))
        print("#test:",   len(test_dataset))
        print("#labels:", len(dataset.label_mapping))
        print("n_per_label:", n_per_label)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,  shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, drop_last=False, num_workers=num_workers)

        model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
        model.load_from_pretrain(pretrain_path)
        model.to("cuda")
        task=ClassificationTask(model,args,n_per_label,classes_num)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="best_models/",
            filename=f"sample-fold{fold:02d}"+"-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
            )

        best_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="best_models/",
            filename=f"best_model-fold{fold:02d}",
            save_top_k=1,
            mode="min",
            )


        #trainer = pl.Trainer(gpus=1,log_every_n_steps=1,max_epochs=10,enable_progress_bar=False)
        trainer = pl.Trainer.from_argparse_args(args,callbacks=[checkpoint_callback,best_checkpoint_callback])
        trainer.fit(task, train_dataloaders=train_loader,val_dataloaders=valid_loader)
        print("==test==")
        trainer.test(task, dataloaders=[test_loader])

        ### loss plot
        print(task.train_losses)
        print(task.val_losses)
        with open("loss_log_fold{:02d}.tsv".format(fold),"w") as ofp:
            for l1,l2 in itertools.zip_longest(task.train_losses,task.val_losses,fillvalue=''):
                ofp.write(str(l1)+"\t"+str(l2)+"\n")

        ### save best model
        model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
        print(f"[load] best_models/best_model-fold{fold:02d}.ckpt")
        task=ClassificationTask.load_from_checkpoint(checkpoint_path=f"best_models/best_model-fold{fold:02d}.ckpt",model=model,args=args)
        task.samples_per_cls=n_per_label
        task.no_of_classes=classes_num
        """
        print("=====")
        print(task.samples_per_cls)
        print(task.no_of_classes)
        print("=====")
        """

        trainer = pl.Trainer.from_argparse_args(args)
        trainer.test(task, dataloaders=[test_loader])
        model_path=f"best_models/best_model-fold{fold:02d}.pth"
        torch.save(trainer.model.model.to('cpu').state_dict(), model_path)

        # evaluation
        model.eval()
        pred_y_prob_all=[]
        pred_y_all=[]
        y_all=[]
        for x,y in test_loader:
            with torch.no_grad():   
                pred_y_prob = model(x).to('cpu').detach().numpy().copy()
                #print(pred_y_prob)
                #print(pred_y_prob.shape)
                pred_y = np.argmax(pred_y_prob,axis=1)
                pred_y_all.extend(pred_y)
                pred_y_prob_all.extend(pred_y_prob)
                y_all.extend(y.to('cpu').detach().numpy())
        acc = accuracy_score(y_all, pred_y_all)
        print("accuracy:",acc)
        for i,idx in enumerate(test_index):
            result.append([fold,idx,y_all[i],pred_y_all[i], pred_y_prob_all[i]])
        print("====")
    print("[save] result_cv.tsv")
    with open("result_cv.tsv","w") as ofp:
        for line in result:
            ofp.write("\t".join(map(str,line[:4])))
            ofp.write("\t")
            ofp.write("\t".join(map(str,line[4])))
            ofp.write("\n")

if __name__ == "__main__":
    args = get_args()
    if args.mode=="train":
        train(args)
    elif args.mode=="train_cv":
        train_cv(args)
    elif args.mode in ["pred","test"]:
        pred(args)
    else:
        print("Error: unknown mode:",args.mode)

