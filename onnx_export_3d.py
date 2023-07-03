import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pdb
import cv2
import numpy as np
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import torch.nn as nn
import torch
from easydict import EasyDict
import os
import torch
from pathlib import Path
import yaml
from model_3d import *


if __name__ == '__main__':
    device = torch.device('cpu')
    
    ckpt_path = 'ckpt_3d/exp1_crop_320_400_resize_182_182_mask_red_ball/epoch=87-train_loss=0.343-val_loss=0.350-train_acc=0.998-val_acc=0.996.ckpt'
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state['state_dict']
    with open(os.path.join(Path(ckpt_path).parent, 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    model = torch.hub.load('facebookresearch/pytorchvideo', config.model.version, pretrained=True)
    model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=3, bias=True)

    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.replace('model.', '')] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    # torch.save(model.state_dict(), 'ckpt/exp4_ce_loss_less_regularized_cropped_data_320_128/epoch36.pt')

    imgs = torch.randn(1, 3, 9, 182, 182, dtype=torch.float)
    # normalize_vid = NormalizeVideo(
    #     mean = [0.45, 0.45, 0.45],
    #     std = [0.225, 0.225, 0.225]
    # )
    # imgs = np.load('/data2/tungtx2/datn/main_app/cropped_frames.npy')
    # imgs = [cv2.resize(img, (182, 182)) for img in imgs]
    # imgs = np.stack(imgs, axis=0)
    # imgs = imgs.transpose(3, 0, 1, 2)    # shape 3 x n_frames x 182 x 182
    # imgs = imgs / 255.
    # imgs = normalize_vid(torch.from_numpy(imgs)).numpy()
    # imgs = torch.from_numpy(imgs).unsqueeze(0).to(torch.float)
    # out = model(imgs)
    # print('output shape: ', out.shape)
    torch.onnx.export(
        model,
        # {
        #     'imgs': imgs,
        # },
        imgs,
        'ckpt_3d/exp1_crop_320_400_resize_182_182_mask_red_ball/epoch87_mask_red_ball.onnx',
        input_names=['imgs'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={
            "imgs": {0: "batch_size"},
        }
    )
