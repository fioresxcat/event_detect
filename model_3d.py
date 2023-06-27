import torch
torch.set_float32_matmul_precision('high')

import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from model import MyAccuracy

# class X3DModel(nn.Module):
#     def __init__(self, version):
#         super().__init__()
#         model = torch.hub.load('facebookresearch/pytorchvideo', version, pretrained=True)

#         self.blocks = model.blocks[:-1]
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, 1)),
#         )

class X3DModule(pl.LightningModule):
    def __init__(
        self, 
        version, 
        n_classes,
        class_weight: List[float],
        learning_rate: float,
        reset_optimizer: bool,
        pos_weight: float,
        ev_diff_thresh: float
    ):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', version, pretrained=True)
        self.model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.reset_optimizer = reset_optimizer
        self.pos_weight = pos_weight
        self.ev_diff_thresh = ev_diff_thresh
        self.class_weight = class_weight
        self._init_loss_and_metrics()


    def _init_loss_and_metrics(self):
        self.train_acc = MyAccuracy(self.ev_diff_thresh)
        self.val_acc = MyAccuracy(self.ev_diff_thresh)
        self.test_acc = MyAccuracy(self.ev_diff_thresh)

        self.predict_acc = MyAccuracy(self.ev_diff_thresh)
        self.preds, self.labels = torch.empty(size=(0, 3)), torch.empty(size=(0, 3))


    def compute_logits_and_losses(self, imgs, pos, labels):
        logits = self.model(imgs)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=torch.tensor(self.class_weight, device=self.device),
        )
        return logits, loss


    def forward(self, imgs):
        return self.model(imgs)
    

    def step(self, batch, batch_idx, split):
        imgs, pos, labels = batch
        logits, loss = self.compute_logits_and_losses(imgs, pos, labels)

        acc = getattr(self, f'{split}_acc')
        acc(logits, labels)

        # if split in ['val', 'test']:
        #     probs = torch.softmax(logits, dim=1)
        #     for prob, label in zip(probs, labels):
        #         print('probs:', torch.round(input=prob, decimals=2), 'label:', label)

        self.log_dict({
            f'{split}_loss': loss,
            f'{split}_acc': acc,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.trainer.callbacks[0].mode,
            factor=0.2,
            patience=10,
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.trainer.callbacks[0].monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }
    

    def on_fit_start(self) -> None:
        if self.reset_optimizer:
            opt = type(self.trainers.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    import pdb

    with open('config_3d.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    model = X3DModule(**config.model)
    model.eval().to('cuda')
    imgs = torch.randn(2, 3, 9, 182, 182).to('cuda')
    out = model(imgs)
    print(out.shape)
    pdb.set_trace()