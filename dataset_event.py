import numpy as np
from pathlib import Path
import os
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



def load_from_pickle(fp):
    with open(fp, 'rb') as f:
        bin = f.read()
        obj = pickle.loads(bin)
    return obj


def generate_heatmap(size, center, radius):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    source: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    """
    width, height = size
    x0, y0 = center
    radius_x, radius_y = radius

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:,np.newaxis]
    heatmap = np.exp(-4*np.log(2) * ((x-x0)**2/radius_x**2 + (y-y0)**2/radius_y**2))
    return heatmap


def mask_ball_in_img(img: np.array, normalized_pos: tuple, r: tuple):
    h, w = img.shape[:2]
    pos = (int(normalized_pos[0] * w), int(normalized_pos[1] * h))
    img[pos[1]-r[1] : pos[1]+r[1], pos[0]-r[0] : pos[0]+r[0], :] = 0
    return img



class EventDataset(Dataset):
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
        super(EventDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.do_augment = do_augment
        self.augment_props = augment_props
        self.orig_h, self.orig_w = 1080, 1920
        self.n_input_frames = n_input_frames
        self.n_sample_limit = int(n_sample_limit)
        self.crop_size = crop_size
        self.input_size = input_size
        self.mask_red_ball = mask_red_ball
        self.ball_radius = ball_radius
        self.thickness = thickness
        self.already_cropped = already_cropped
        self.transforms = transforms
        self.tab_xmax_offset = tab_xmax_offset
        self.jpeg_reader = TurboJPEG()

        self._init_paths_and_labels()
    

    def _init_paths_and_labels(self):
        data_dict = load_from_pickle(self.data_path)
        self.ls_img_paths = []
        self.ls_labels = []
        for img_paths, labels in data_dict.items():
            self.ls_img_paths.append(sorted(img_paths))
            self.ls_labels.append(labels)
        self.ls_img_paths = self.ls_img_paths[:int(self.n_sample_limit)]

        # # shuffle
        # np.random.seed(42)
        # indices = np.arange(len(self.ls_img_paths))
        # np.random.shuffle(indices)
        # self.ls_img_paths = [self.ls_img_paths[i] for i in indices]
        # self.ls_labels = [self.ls_labels[i] for i in indices]


    def __len__(self):
        return len(self.ls_img_paths)
    

    def crop_images_from_paths(self, img_paths, ls_norm_pos, event_target):
        # mask pos
        if self.mode == 'train' and self.do_augment and np.random.rand() < self.augment_props.mask_ball_prob:
            if np.random.rand() < 1:  # random mask
                num_invalid_pos = len([pos for pos in ls_norm_pos if pos[0]< 0 or pos[1] < 0])
                max_size = self.augment_props.max_mask_ball - num_invalid_pos
                mask_indices = np.random.choice(list(range(len(ls_norm_pos))), size=np.random.randint(0, max_size+1))
                for idx in mask_indices:
                    ls_norm_pos[idx] = (-1, -1)

            # else:   # mask first half or second half
            #     first_part_pos = [pos for i, pos in enumerate(ls_norm_pos) if i <= len(ls_norm_pos)//2]
            #     second_part_pos = [pos for i, pos in enumerate(ls_norm_pos) if i >= len(ls_norm_pos)//2]
            #     first_pos_valid = all(pos[0] > 0 and pos[1] > 0 for pos in first_part_pos)
            #     second_pos_valid = all(pos[0] > 0 and pos[1] > 0 for pos in second_part_pos)
            #     if first_pos_valid and second_pos_valid:
            #         if np.random.rand() < 0.5:
            #             for i in range(len(ls_norm_pos)//2+1, len(ls_norm_pos)):
            #                 ls_norm_pos[i] = (-1, -1)
            #         else:
            #             for i in range(0, len(ls_norm_pos)//2):
            #                 ls_norm_pos[i] = (-1, -1)
            #     elif first_pos_valid:
            #         for i in range(len(ls_norm_pos)//2+1, len(ls_norm_pos)):
            #             ls_norm_pos[i] = (-1, -1)
            #     elif second_pos_valid:
            #         for i in range(0, len(ls_norm_pos)//2):
            #             ls_norm_pos[i] = (-1, -1)

        
        # select crop coords
        ls_valid_pos = [pos for pos in ls_norm_pos if pos[0] > 0 and pos[1] > 0]
        median_cx = np.median([pos[0] for pos in ls_valid_pos])
        median_cy = np.median([pos[1] for pos in ls_valid_pos])
        median_cx = int(median_cx*1920)
        median_cy = int(median_cy*1080)
        xmin = max(0, median_cx - self.crop_size[0]//2)
        xmax = min(median_cx + self.crop_size[0]//2, 1920)
        ymin = max(0, median_cy - self.crop_size[1] // 3)   # crop only 1/3 on top
        ymax = min(median_cy + self.crop_size[1]*2//3, 1080)    # crop 2/3 on bottom

        # crop imgs
        input_imgs = []
        for i, fp in enumerate(img_paths):
            with open(fp, 'rb') as in_file:
                orig_img = self.jpeg_reader.decode(in_file.read(), 0)
            if self.mask_red_ball:
                pos = ls_norm_pos[i]
                if pos[0] > 0 and pos[1] > 0:
                    abs_pos = (int(pos[0] * orig_img.shape[1]), int(pos[1] * orig_img.shape[0]))
                    orig_img = cv2.circle(orig_img, abs_pos, self.ball_radius, (255, 0, 0), thickness=self.thickness)

            cropped_img = orig_img[ymin:ymax, xmin:xmax]
            # resize
            cropped_img = cv2.resize(cropped_img, self.input_size)

            # append
            input_imgs.append(cropped_img)
        
        # out_dir = 'test'
        # os.makedirs(out_dir, exist_ok=True)
        # for i, img in enumerate(input_imgs):
        #     img_fn = Path(img_paths[i]).parent.name + '_' + Path(img_paths[i]).name
        #     cv2.imwrite(f'{out_dir}/{img_fn}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # pdb.set_trace()

        return input_imgs, ls_norm_pos
    

    def get_already_cropped_images(self, img_paths, ls_norm_pos, event_target):
        input_imgs = []
        if self.mode == 'train' and self.do_augment and np.random.rand() < self.augment_props.mask_ball_prob:
            num_invalid_pos = len([pos for pos in ls_norm_pos if pos[0]< 0 or pos[1] < 0])
            max_size = self.augment_props.max_mask_ball - num_invalid_pos
            mask_indices = np.random.choice(list(range(len(ls_norm_pos))), size=np.random.randint(0, max_size+1))
        else:
            mask_indices = []
        for i, img_fp in enumerate(img_paths):
            pos = ls_norm_pos[i]
            if pos[0] < 0 or pos[1] < 0 or i in mask_indices:
                ls_norm_pos[i] = (-1, -1)
            with open(img_fp, 'rb') as in_file:
                cropped_img = self.jpeg_reader.decode(in_file.read(), 0)  # already rgb images
            # pdb.set_trace()
            cropped_img = cv2.resize(cropped_img, self.crop_size)
            input_imgs.append(cropped_img)
        
        return input_imgs, ls_norm_pos


    def get_masked_images_old(self, img_paths, labels):
        ls_norm_pos, ev_target, ev_type, mask_fp = labels

        if self.mode == 'train' and self.do_augment and np.random.rand() < self.augment_props.mask_ball_prob:
            num_invalid_pos = len([pos for pos in ls_norm_pos if pos[0]< 0 or pos[1] < 0])
            max_size = max(self.augment_props.max_mask_ball - num_invalid_pos, 0)
            mask_indices = np.random.choice(list(range(len(ls_norm_pos))), size=np.random.randint(0, max_size+1))
            for idx in mask_indices:
                ls_norm_pos[idx] = (-1, -1)

        # select crop coords
        ls_valid_pos = [pos for pos in ls_norm_pos if pos[0] > 0 and pos[1] > 0]
        median_cx = np.median([pos[0] for pos in ls_valid_pos])
        median_cy = np.median([pos[1] for pos in ls_valid_pos])
        median_cx = int(median_cx*1920)
        median_cy = int(median_cy*1080)
        xmin = max(0, median_cx - self.crop_size[0]//2)
        xmax = min(median_cx + self.crop_size[0]//2, 1920)
        ymin = max(0, median_cy - self.crop_size[1] // 5)   # crop only 1/3 on top
        ymax = min(median_cy + self.crop_size[1]*4//5, 1080)    # crop 2/3 on bottom

        # process mask
        mask = cv2.imread(mask_fp)
        tab_mask = mask[:, :, 2]

        contours, hierarchy = cv2.findContours(tab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[0]
        tab_xmin, tab_ymin, w, h = cv2.boundingRect(cnt)
        tab_xmax = tab_xmin + w
        tab_ymax = tab_ymin + h
        tab_xmin = int(tab_xmin / tab_mask.shape[1] * 1920)
        tab_xmax = int(tab_xmax / tab_mask.shape[1] * 1920)
        tab_ymin = int(tab_ymin / tab_mask.shape[0] * 1080)
        tab_ymax = int(tab_ymax / tab_mask.shape[0] * 1080)

        tab_mask = cv2.resize(tab_mask, (1920, 1080))
        tab_mask[ymin:ymax, xmin:xmax] = 255
        tab_mask = cv2.cvtColor(tab_mask, cv2.COLOR_GRAY2RGB)
        
        # crop
        crop_xmin = min(tab_xmin, xmin)
        crop_xmax = max(tab_xmax, xmax)
        crop_ymin = min(tab_ymin, ymin)
        crop_ymax = tab_ymax


        input_imgs = []
        for i, fp in enumerate(img_paths):
            with open(fp, 'rb') as in_file:
                orig_img = self.jpeg_reader.decode(in_file.read(), 0)
            
            if self.mask_red_ball:
                pos = ls_norm_pos[i]
                if pos[0] > 0 and pos[1] > 0:
                    abs_pos = (int(pos[0] * orig_img.shape[1]), int(pos[1] * orig_img.shape[0]))
                    orig_img = cv2.circle(orig_img, abs_pos, self.ball_radius, (255, 0, 0), thickness=self.thickness)

            # mask image
            masked_img = np.where(tab_mask > 0, orig_img, 0)
            masked_img = masked_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            # resize
            # masked_img = cv2.resize(masked_img, self.input_size)
            input_imgs.append(masked_img)

            # if 'test_1' in fp:
            # cv2.imwrite('a.png', masked_img)
            # pdb.set_trace()

        
        return input_imgs, ls_norm_pos


    def get_masked_images(self, img_paths, labels):
        ls_norm_pos, ev_target, ev_type, mask_fp = labels

        if self.mode == 'train' and self.do_augment and np.random.rand() < self.augment_props.mask_ball_prob:
            num_invalid_pos = len([pos for pos in ls_norm_pos if pos[0]< 0 or pos[1] < 0])
            max_size = max(self.augment_props.max_mask_ball - num_invalid_pos, 0)
            mask_indices = np.random.choice(list(range(len(ls_norm_pos))), size=np.random.randint(0, max_size+1))
            for idx in mask_indices:
                ls_norm_pos[idx] = (-1, -1)

        # select crop coords
        ls_valid_pos = [pos for pos in ls_norm_pos if pos[0] > 0 and pos[1] > 0]
        median_cx = np.median([pos[0] for pos in ls_valid_pos])
        median_cy = np.median([pos[1] for pos in ls_valid_pos])
        median_cx = int(median_cx*1920)
        median_cy = int(median_cy*1080)
        xmin = max(0, median_cx - self.crop_size[0]//2)
        xmax = min(median_cx + self.crop_size[0]//2, 1920)
        ymin = max(0, median_cy - self.crop_size[1] // 2)   # crop only 1/3 on top
        ymax = min(median_cy + self.crop_size[1]*1//2, 1080)    # crop 2/3 on bottom


        # process mask
        mask = cv2.imread(mask_fp)
        tab_mask = mask[:, :, 2]
        contours, hierarchy = cv2.findContours(tab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[0]
        tab_xmin, tab_ymin, w, h = cv2.boundingRect(cnt)
        tab_xmax = tab_xmin + w
        tab_ymax = tab_ymin + h
        tab_xmin = int(tab_xmin / tab_mask.shape[1] * 1920)
        tab_xmax = int(tab_xmax / tab_mask.shape[1] * 1920)
        tab_ymin = int(tab_ymin / tab_mask.shape[0] * 1080)
        tab_ymax = int(tab_ymax / tab_mask.shape[0] * 1080)
            

        input_imgs = []
        for i, fp in enumerate(img_paths):
            with open(fp, 'rb') as in_file:
                orig_img = self.jpeg_reader.decode(in_file.read(), 0)
            
            if self.mask_red_ball:
                pos = ls_norm_pos[i]
                if pos[0] > 0 and pos[1] > 0:
                    abs_pos = (int(pos[0] * orig_img.shape[1]), int(pos[1] * orig_img.shape[0]))
                    orig_img = cv2.circle(orig_img, abs_pos, self.ball_radius, (255, 0, 0), thickness=self.thickness)

            # crop ball part
            ball_part = orig_img[ymin:min(ymax, tab_ymin), xmin:xmax]
            # crop table part
            table_part = orig_img[tab_ymin:tab_ymax, xmin:xmax]
            # concat
            concat_img = np.concatenate((ball_part, table_part), axis=0)
            concat_img = cv2.resize(concat_img, self.input_size)
            input_imgs.append(concat_img)

            # if 'test_1' in fp:
            # cv2.imwrite('a.png', masked_img)
            # pdb.set_trace()

        
        return input_imgs, ls_norm_pos
    

    def get_masked_images_new(self, img_paths, labels):
        ls_norm_pos, ev_target, ev_type, mask_fp, tab_bb_fp = labels

        if self.mode == 'train' and self.do_augment and np.random.rand() < self.augment_props.mask_ball_prob:
            if np.random.rand() < self.augment_props.mask_ball_prob:
                num_invalid_pos = len([pos for pos in ls_norm_pos if pos[0]< 0 or pos[1] < 0])
                max_size = max(self.augment_props.max_mask_ball - num_invalid_pos, 0)
                mask_indices = np.random.choice(list(range(len(ls_norm_pos))), size=np.random.randint(0, max_size+1))
                for idx in mask_indices:
                    ls_norm_pos[idx] = (-1, -1)

        # select crop coords
        ls_valid_pos = [pos for pos in ls_norm_pos if pos[0] > 0 and pos[1] > 0]
        median_cx = np.median([pos[0] for pos in ls_valid_pos])
        median_cy = np.median([pos[1] for pos in ls_valid_pos])
        median_cx = int(median_cx*1920)
        median_cy = int(median_cy*1080)
        xmin = max(0, median_cx - self.crop_size[0]//2)
        xmax = min(median_cx + self.crop_size[0]//2, 1920)
        ymin = max(0, median_cy - self.crop_size[1] // 2)   # crop only 1/3 on top
        ymax = min(median_cy + self.crop_size[1]*1//2, 1080)    # crop 2/3 on bottom

        with open(tab_bb_fp, 'r') as f:
            lines = f.readlines()
        line = lines[0]
        bb = [int(float(el)) for el in line.strip().split()[1:]]
        tab_xmin, tab_ymin, tab_xmax, tab_ymax = bb
        if all(el==0 for el in bb):
            # process mask
            mask = cv2.imread(mask_fp)
            tab_mask = mask[:, :, 2]
            contours, hierarchy = cv2.findContours(tab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cnt = contours[0]
            tab_xmin, tab_ymin, w, h = cv2.boundingRect(cnt)
            tab_xmax = tab_xmin + w
            tab_ymax = tab_ymin + h
            tab_xmin = int(tab_xmin / tab_mask.shape[1] * 1920)
            tab_xmax = int(tab_xmax / tab_mask.shape[1] * 1920)
            tab_ymin = int(tab_ymin / tab_mask.shape[0] * 1080)
            tab_ymax = int(tab_ymax / tab_mask.shape[0] * 1080)
            

        input_imgs = []
        for i, fp in enumerate(img_paths):
            with open(fp, 'rb') as in_file:
                orig_img = self.jpeg_reader.decode(in_file.read(), 0)
            
            if self.mask_red_ball:
                pos = ls_norm_pos[i]
                if pos[0] > 0 and pos[1] > 0:
                    abs_pos = (int(pos[0] * orig_img.shape[1]), int(pos[1] * orig_img.shape[0]))
                    orig_img = cv2.circle(orig_img, abs_pos, self.ball_radius, (255, 0, 0), thickness=self.thickness)

            # crop ball part
            ball_part = orig_img[ymin:min(ymax, tab_ymin), xmin:xmax]
            # crop table part
            table_part = orig_img[tab_ymin:tab_ymax+self.tab_xmax_offset, xmin:xmax]
            # concat
            concat_img = np.concatenate((ball_part, table_part), axis=0)
            concat_img = cv2.resize(concat_img, self.input_size)
            input_imgs.append(concat_img)

            # cv2.imwrite('a.png', concat_img)
            # pdb.set_trace()

        
        return input_imgs, ls_norm_pos
    

    def __getitem__(self, index):
        img_paths = self.ls_img_paths[index]
        labels = self.ls_labels[index]
        ls_norm_pos, event_target, ev_type, mask_fp = labels

        input_imgs, ls_norm_pos = self.get_masked_images_new(img_paths, labels)

        if self.mode == 'train' and np.random.rand() < self.augment_props.augment_img_prob:
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
            # for i, img in enumerate(transformed_imgs):
            #     cv2.imwrite('a.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            #     print(f'just save image {i}')
            #     pdb.set_trace()

            transformed_imgs = np.concatenate(transformed_imgs, axis=2)
            transformed_imgs = torch.tensor(transformed_imgs)
        else:
            transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))

        # normalize
        transformed_imgs = transformed_imgs.permute(2, 0, 1) / 255.

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




class EventDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path, 
        test_path,
        predict_path,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(EventDataModule, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.data_cfg = EasyDict(data_cfg)
        self.training_cfg = EasyDict(training_cfg)

        if self.data_cfg.do_augment:
            if self.data_cfg.n_input_frames == 3:
                add_target = {'image0': 'image', 'image1': 'image'}
            elif self.data_cfg.n_input_frames == 5:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}
            elif self.data_cfg.n_input_frames == 9:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image'}
            
            # self.transforms = A.Compose(
            #     A.SomeOf([
            #         A.HorizontalFlip(p=0.5),
            #         A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
            #         A.ColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
            #         A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=0),
            #     ], n=2),
            #     additional_targets=add_target,
            # )

            self.transforms = A.Compose(
                A.SomeOf([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.6, shift_limit=0.1, scale_limit=0.12, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.ColorJitter(p=0.6, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
                    A.Perspective(p=0.4, scale=(0.03, 0.07), keep_size=True, fit_output=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),
                    A.SafeRotate(p=0.5, limit=3, border_mode=cv2.BORDER_CONSTANT, value=0),
                    # A.Affine(p=1, scale=(0.8, 1.15), translate_percent=(0, 0.1), rotate=(-7, 7), shear=(-10, 10), mode=cv2.BORDER_REPLICATE)
                    # A.ElasticTransform(p=1, alpha=1, sigma=50, alpha_affine=30, border_mode=cv2.BORDER_REPLICATE, value=0),
                ], n=3),
                additional_targets=add_target,
            )
        else:
            self.transforms = None

    
    def setup(self, stage):
        if stage == 'fit' or stage == 'validate':
            self.train_ds = EventDataset(data_path=self.train_path, transforms=self.transforms, mode='train', **self.data_cfg)
            self.val_ds = EventDataset(data_path=self.val_path, transforms=None, mode='val', **self.data_cfg)
        elif stage == 'test':
            self.test_ds = EventDataset(self.test_path, transforms=None, mode='test', **self.data_cfg)
        elif stage == 'predict':
            self.predict_ds = EventDataset(self.predict_path, transforms=None, mode='predict', **self.data_cfg)

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

    config.data.training_cfg.num_workers = 0
    ds_module = EventDataModule(**config.data)
    ds_module.setup('fit')

    for i, item in enumerate(ds_module.train_ds):
        imgs, pos, ev = item
        print(imgs.shape)
        print(pos.shape)
        print(ev.shape)
        # break
    pdb.set_trace()
    print('ok')