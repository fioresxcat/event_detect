import numpy as np
import pdb
from easydict import EasyDict
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import torch
import pytorch_lightning as pl
from PIL import Image
import time
from turbojpeg import TurboJPEG
import albumentations as A
from dataset_event import EventDataset, EventDataModule
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

class EventDataset3D(EventDataset):
    def __init__(
        self,
        data_path,
        transforms, 
        mode,
        n_input_frames,
        n_sample_limit,
        crop_size,
        input_size,
        mask_red_ball,
        ball_radius,
        thickness,
        tab_xmax_offset,
        already_cropped,
        do_augment=False,
        augment_props={},
    ):
        super(EventDataset3D, self).__init__(data_path, transforms, mode, n_input_frames, n_sample_limit, crop_size, input_size, mask_red_ball, ball_radius, thickness, tab_xmax_offset, already_cropped, do_augment, augment_props)
        self.normalize_video = NormalizeVideo(
            mean = [0.45, 0.45, 0.45],
            std = [0.225, 0.225, 0.225]
        )


    def __getitem__(self, index):
        img_paths = self.ls_img_paths[index]
        labels = self.ls_labels[index]
        ls_norm_pos, event_target, ev_type, mask_fp, tab_bb_fp = labels
        input_imgs, ls_norm_pos = self.get_masked_images_new(img_paths, labels)
        
        if self.mode == 'train' and self.do_augment and np.random.rand() < self.augment_props.augment_img_prob:
            transformed = self.transforms(
                image=input_imgs[0],
                image0=input_imgs[1],
                image1=input_imgs[2],
                image2=input_imgs[3],
                image3=input_imgs[4],
                image4=input_imgs[5],
                image5=input_imgs[6],
                image6=input_imgs[7],
                image7=input_imgs[8],
            )
            transformed_imgs = [transformed[k] for k in sorted([k for k in transformed.keys() if k.startswith('image')])]
            transformed_imgs = np.stack(transformed_imgs, axis=0)
        else:
            transformed_imgs = np.stack(input_imgs, axis=0)   # shape 9 x h x w x 3

        # for i, img in enumerate(transformed_imgs):
        #     cv2.imwrite('a.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #     print(f'just save image {i}')
        #     pdb.set_trace()


        # normalize
        transformed_imgs = torch.from_numpy(transformed_imgs)
        transformed_imgs = transformed_imgs.permute(3, 0, 1, 2) # shape 3 x 9 x h x w
        transformed_imgs = transformed_imgs / 255.0
        transformed_imgs = self.normalize_video(transformed_imgs)

        # construct event target
        if event_target[0] != 0:
            event_target = torch.tensor([event_target[0], 0, 1-event_target[0]], dtype=torch.float)
        elif event_target[1] != 0:
            event_target = torch.tensor([0, event_target[1], 1-event_target[1]], dtype=torch.float)
        else:
            event_target = torch.tensor([0, 0, 1], dtype=torch.float)

        if self.mode != 'predict':
            return transformed_imgs, torch.tensor(ls_norm_pos), event_target
        else:
            return img_paths, transformed_imgs, torch.tensor(ls_norm_pos), event_target
        

class EventDataModule3D(EventDataModule):
    def __init__(
        self,
        train_path,
        val_path, 
        test_path,
        predict_path,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(EventDataModule3D, self).__init__(train_path, val_path, test_path, predict_path, data_cfg, training_cfg)
    

    def setup(self, stage):
        if stage == 'fit' or stage == 'validate':
            self.train_ds = EventDataset3D(data_path=self.train_path, transforms=self.transforms, mode='train', **self.data_cfg)
            self.val_ds = EventDataset3D(data_path=self.val_path, transforms=None, mode='val', **self.data_cfg)
        elif stage == 'test':
            self.test_ds = EventDataset3D(self.test_path, transforms=None, mode='test', **self.data_cfg)
        elif stage == 'predict':
            self.predict_ds = EventDataset3D(self.predict_path, transforms=None, mode='predict', **self.data_cfg)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=self.training_cfg.shuffle_train, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    with open('config_3d.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.data.training_cfg.num_workers = 0

    ds_module = EventDataModule3D(**config.data)
    ds_module.setup('fit')
    ds = ds_module.train_ds

    for i, item in enumerate(ds):
        imgs, pos, ev = item
        print(imgs.shape)
        print(pos.shape)
        # print('img paths: ', ds.ls_img_paths[i])
        # print('ev_probs: ', ev)
        # break
    pdb.set_trace()
    print('ok')