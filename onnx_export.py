import pdb
from easydict import EasyDict
import os
import torch
from pathlib import Path
import yaml
from model import *

if __name__ == '__main__':
    ckpt_path = 'ckpt/exp4_ce_loss_less_regularized_cropped_data_320_128/epoch=36-train_loss=0.008-val_loss=0.087-train_acc=0.995-val_acc=0.978.ckpt'
    state = torch.load(ckpt_path)
    state_dict = state['state_dict']
    with open(os.path.join(Path(ckpt_path).parent, 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    model = EventClassifierModel(
        cnn_cfg=config.model.model.init_args.cnn_cfg,
        lstm_cfg=config.model.model.init_args.lstm_cfg,
        classifier_dropout=config.model.model.init_args.classifier_dropout,
        num_classes=config.model.model.init_args.num_classes
    )
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.replace('model.', '')] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    torch.save(model.state_dict(), 'ckpt/exp4_ce_loss_less_regularized_cropped_data_320_128/epoch36.pt')

    imgs = torch.rand(1, 27, 128, 320)
    pos = torch.rand(1, 9, 2)
    
    torch.onnx.export(
        model,
        {
            'imgs': imgs,
            'pos': pos
        },
        'ckpt/exp4_ce_loss_less_regularized_cropped_data_320_128/epoch36.onnx',
        input_names=['imgs', 'pos'],
        output_names=['output'],
        opset_version=14
    )
