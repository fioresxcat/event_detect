from typing import Any, List
from easydict import EasyDict
import pdb
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import pytorch_lightning as pl


# class MyAccuracy(torchmetrics.Metric):
#     # Set to True if the metric reaches it optimal value when the metric is maximized.
#     # Set to False if it when the metric is minimized.
#     higher_is_better = True

#     # Set to True if the metric during 'update' requires access to the global metric
#     # state for its calculations. If not, setting this to False indicates that all
#     # batch states are independent and we will optimize the runtime of 'forward'
#     full_state_update = True

#     def __init__(self, ev_diff_thresh):
#         super().__init__()
#         self.ev_diff_thresh = ev_diff_thresh
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         diff = torch.abs(torch.sigmoid(preds) - target)   # shape nx3, event_pred is not sigmoided inside the model
#         n_true = (diff < self.ev_diff_thresh).all(dim=1).sum()

#         self.correct += n_true
#         self.total += target.shape[0]

#     def compute(self):
#         return self.correct.float() / self.total




class MyAccuracy_2(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self, ev_diff_thresh):
        super().__init__()
        self.ev_diff_thresh = ev_diff_thresh
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        n_true = 0
        for i in range(len(preds)):
            pred = preds[i]
            true = target[i]
            max_pred_value, max_pred_idx = torch.max(pred, dim=0)
            max_true_value, max_true_idx = torch.max(true, dim=0)
            if max_pred_idx == max_true_idx:
                n_true += 1
            elif max_true_value < 0.5 and max_true_idx != 2:
                if max_pred_idx == 2:
                    n_true += 1
            

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total


class MyAccuracy_3(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self, ev_diff_thresh):
        super().__init__()
        self.ev_diff_thresh = ev_diff_thresh
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = torch.softmax(preds, dim=1)
        preds = torch.sigmoid(preds)

        # filter only rows that have 1 in its values
        preds = preds[(target==1).any(dim=1)]
        target = target[(target==1).any(dim=1)]
        print(preds)
        print(target)
        max_pred_indices = torch.argmax(preds, dim=1)
        max_true_indices = torch.argmax(target, dim=1)
        n_true = (max_pred_indices==max_true_indices).sum()

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total



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
        reset_optimizer: bool,
        pos_weight: float,
        ev_diff_thresh: float
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.reset_optimizer = reset_optimizer
        self.pos_weight = pos_weight
        self.ev_diff_thresh = ev_diff_thresh
        self._init_losses_and_metrics()

    
    def _init_losses_and_metrics(self):        
        # self.train_total, self.train_tp, self.train_tn, self.train_fp, self.train_fn = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        # self.val_total, self.val_tp, self.val_tn, self.val_fp, self.val_fn = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        # self.test_total, self.test_tp, self.test_tn, self.test_fp, self.test_fn = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        # self.train_true, self.train_total = 0, 0
        # self.val_true, self.val_total = 0, 0
        # self.test_true, self.test_total = 0, 0

        self.train_acc = MyAccuracy(self.ev_diff_thresh)
        self.val_acc = MyAccuracy(self.ev_diff_thresh)
        self.test_acc = MyAccuracy(self.ev_diff_thresh)

        self.predict_acc = MyAccuracy_3(self.ev_diff_thresh)

        self.preds, self.labels = torch.empty(size=(0, 3)), torch.empty(size=(0, 3))



    def _compute_loss_and_outputs(self, imgs, pos, labels):
        logits = self.model(imgs, pos)

        loss = F.cross_entropy(
            logits, # shape n x 3
            labels,  # shape n x 3
            weight=torch.tensor([1, 1, 1], device=self.device)  # empty event xuất hiện ít hơn -> weight cao hơn
        )

        # loss = F.binary_cross_entropy_with_logits(
        #     logits,
        #     labels,
        # )

        return loss, logits
    

    def step(self, batch, batch_idx, split):
        imgs, pos, labels = batch
        loss, logits = self._compute_loss_and_outputs(imgs, pos, labels)

        acc = getattr(self, f'{split}_acc')
        acc(logits, labels)

        self.log_dict({
            f'{split}_loss': loss,
            f"{split}_acc": acc,
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    

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



    def on_fit_start(self) -> None:
        if self.reset_optimizer:
            opt = type(self.trainers.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')


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