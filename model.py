from typing import Any, List
from easydict import EasyDict
import pdb
from copy import deepcopy
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import pytorch_lightning as pl
from metric import *
import os
import cv2
from pathlib import Path
import numpy as np



class LSTMModel(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=16, num_layers=2, output_size=16, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    

class EventClassifierModel(pl.LightningModule):
    def __init__(self, cnn_cfg, lstm_cfg, classifier_dropout, num_classes):
        super().__init__()
        self.cnn_cfg = EasyDict(cnn_cfg)
        self.lstm_cfg = EasyDict(lstm_cfg)

        effb0 = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        effb0.features[0][0] = nn.Conv2d(3*self.cnn_cfg.num_frames, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        effb0.features = effb0.features[:self.cnn_cfg.cut_index]

        self.cnn = nn.Sequential(
            effb0.features,
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
        )
        self.lstm = LSTMModel(**lstm_cfg)
        self.fc1 = nn.Linear(120, 32)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.fc2 = nn.Linear(32, num_classes)

    
    def forward(self, imgs, pos):
        out_cnn = self.cnn(imgs)    # shape (n x 112)
        out_lstm = self.lstm(pos)   # shape (n x 16)
        fuse = torch.concat([out_cnn, out_lstm], dim=-1)
        x = self.fc1(fuse)
        x = self.act1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out
    

class EventClassifierModule(pl.LightningModule):
    def __init__(
        self,
        model: EventClassifierModel,
        learning_rate: float,
        n_classes: int,
        class_weight: List[float],
        reset_optimizer: bool,
        loss: nn.Module,
        acc: torchmetrics.Metric,
        relaxed_acc: torchmetrics.Metric,
        pce: torchmetrics.Metric,
        smooth_pce: torchmetrics.Metric,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.reset_optimizer = reset_optimizer
        self.n_classes = n_classes
        self.class_weight = class_weight

        # loss and metrics
        self.criterion = loss

        self.train_acc = acc
        self.val_acc = deepcopy(acc)
        self.test_acc = deepcopy(acc)

        self.train_relaxed_acc = relaxed_acc
        self.val_relaxed_acc = deepcopy(relaxed_acc)
        self.test_relaxed_acc = deepcopy(relaxed_acc)

        self.train_pce = pce
        self.val_pce = deepcopy(pce)
        self.test_pce = deepcopy(pce)

        self.train_smooth_pce = smooth_pce
        self.val_smooth_pce = deepcopy(smooth_pce)
        self.test_smooth_pce = deepcopy(smooth_pce)

        self.preds, self.labels = torch.empty(size=(0, self.n_classes)), torch.empty(size=(0, self.n_classes))
    

    def _compute_loss_and_outputs(self, imgs, pos, labels):
        logits = self.model(imgs, pos)

        loss = self.criterion(logits, labels)

        return loss, logits
    

    def step(self, batch, batch_idx, split):
        imgs, pos, labels = batch
        loss, logits = self._compute_loss_and_outputs(imgs, pos, labels)

        acc = getattr(self, f'{split}_acc')
        acc(logits, labels)

        relaxed_acc = getattr(self, f'{split}_relaxed_acc')
        relaxed_acc(logits, labels)

        pce = getattr(self, f'{split}_pce')
        pce(logits, labels)

        smooth_pce = getattr(self, f'{split}_smooth_pce')
        smooth_pce(logits, labels)

        # if split in ['val', 'test']:
        #     probs = torch.softmax(logits, dim=1)
        #     for prob, label in zip(probs, labels):
        #         print('probs:', torch.round(input=prob, decimals=2), 'label:', label)

        self.log_dict({
            f'{split}_loss': loss,
            f'{split}_acc': acc,
            f'{split}_relaxed_acc': relaxed_acc,
            f'{split}_pce': pce,
            f'{split}_smooth_pce': smooth_pce,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    

    def test_step(self, batch, batch_idx):
        imgs, pos, labels = batch
        logits, loss = self.compute_logits_and_losses(imgs, pos, labels)

        acc = getattr(self, f'test_acc')
        acc(logits, labels)

        relaxed_acc = getattr(self, f'test_relaxed_acc')
        relaxed_acc(logits, labels)

        pce = getattr(self, f'test_pce')
        pce(logits, labels)

        smooth_pce = getattr(self, f'test_smooth_pce')
        smooth_pce(logits, labels)


        self.log_dict({
            f'test_loss': loss,
            f'test_acc': acc,
            f'test_relaxed_acc': relaxed_acc,
            f'test_pce': pce,
            f'test_smooth_pce': smooth_pce,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # save wrong case for debug
        bs = self.trainer.datamodule.training_cfg.bs
        all_img_paths = self.trainer.datamodule.test_ds.ls_img_paths
        batch_img_paths = all_img_paths[(bs*batch_idx):(bs*(batch_idx+1))]

        # false_indices, pred_probs, gt_probs, extended_false_indices, extended_pred_probs, extended_gt_probs = acc.get_false_indices(logits, labels)
        false_indices, pred_probs, gt_probs = relaxed_acc.get_false_indices(logits, labels)
        acc_type = 'relaxed_acc'
        
        false_img_paths = [batch_img_paths[i] for i in false_indices]
        ls_ball_pos = pos[false_indices].tolist()

        for img_paths, pred_prob, gt_prob, ball_pos in zip(false_img_paths, pred_probs, gt_probs, ls_ball_pos):
            ls_stem = [Path(img_path).stem for img_path in img_paths]
            ls_frame_idx = [int(stem.split('_')[-1]) for stem in ls_stem]
            min_fr = min(ls_frame_idx)
            max_fr = max(ls_frame_idx)
            game_name = Path(img_paths[0]).parent.stem
            save_dir = os.path.join(self.save_debug_dir, acc_type, game_name, f'{min_fr}-{max_fr}')
            os.makedirs(save_dir, exist_ok=True)
            
            # write probs
            with open(os.path.join(save_dir, 'probs.txt'), 'w') as f:
                f.write(f'pred probs: {pred_prob}\n')
                f.write(f'gt probs: {gt_prob}\n')
            
            abs_pos = np.array(ball_pos) * np.array([1920, 1080])
            abs_pos = abs_pos.astype(int).tolist()

            # write cropped frames
            median_cx = np.median([pos[0] for pos in abs_pos])
            median_cy = np.median([pos[1] for pos in abs_pos])
            xmin = int(max(0, median_cx - self.debug_crop_size[0]//2))
            xmax = int(min(median_cx + self.debug_crop_size[0]//2, 1920))
            ymin = int(max(0, median_cy - self.debug_crop_size[1] // 3))  # crop only 1/3 on top
            ymax = int(min(median_cy + self.debug_crop_size[1]*2//3, 1080))    # crop 2/3 on bottom

            # write original image
            for img_idx, img_fp in enumerate(img_paths):
                img = cv2.imread(str(img_fp))
                img = cv2.circle(img, tuple(abs_pos[img_idx]), 20, (0, 0, 255), 3)
                resized = cv2.resize(img, (720, 480))
                cv2.imwrite(os.path.join(save_dir, Path(img_fp).name), resized)

                cropped = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(save_dir, f'cropped_{Path(img_fp).name}'), cropped)


            with open(os.path.join(save_dir, 'ball.txt'), 'w') as f:
                for i, fr_idx in enumerate(ls_frame_idx):
                    norm_pos = ball_pos[i]
                    abs_pos = (int(norm_pos[0]*1920), int(norm_pos[1]*1080))
                    f.write(f'{fr_idx}: {abs_pos}\n')
    

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.trainer.callbacks[0].mode,
            factor=0.2,
            patience=10,
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     opt,
        #     gamma=0.97,
        # )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.trainer.callbacks[0].monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }


    def on_train_start(self) -> None:
        if self.reset_optimizer:
            opt = type(self.trainer.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')

    
    def on_test_start(self) -> None:
        ckpt_path = self.trainer.ckpt_path
        exp_name = Path(ckpt_path).parent.name
        epoch = Path(ckpt_path).stem.split('-')[0]
        self.save_debug_dir = os.path.join('debug', f'{exp_name}_{epoch}')
        self.debug_crop_size = self.trainer.datamodule.test_ds.crop_size



    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if batch_idx < int(1e9):
            img_paths, imgs, pos, labels = batch
            logits = self.model(imgs, pos)      # shape n x 3
            self.preds = torch.concat([self.preds, logits])
            self.labels = torch.concat([self.labels, labels])

            # self.predict_acc(logits, labels)

            # pdb.set_trace()
            # preds = torch.softmax(logits, dim=1)   # shape n x 3
            preds = torch.sigmoid(logits)   # shape n x 3

            pred_indices = torch.argmax(preds, dim=1)  # shape n x 1
            label_indices = torch.argmax(labels, dim=1) # shape n x 1
            
            preds = preds.round(decimals=2)
            labels = labels.round(decimals=2)
            for pred, label in list(zip(preds, labels)):
                print(pred, label)

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        print(self.predict_acc(self.preds, self.labels))        



if __name__ == '__main__':
    import pdb
    from easydict import EasyDict
    import yaml

    cnn_cfg = EasyDict(dict(
        num_frames=9
    ))
    lstm_cfg = EasyDict(dict(
        input_size=2, 
        hidden_size=16, 
        num_layers=2, 
        output_size=16
    ))

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    model = EventClassifierModel(**config.model.model.init_args)

    pdb.set_trace()
    imgs = torch.rand(2, 27, 128, 128)
    pos = torch.rand(2, 9, 2)
    out = model(imgs, pos)
    pdb.set_trace()
    print(out.shape)