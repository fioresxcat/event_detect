import os
import cv2
from PIL import Image
from pathlib import Path
import json
import pdb
import albumentations as A
import pickle


def get_test_result():
    data_dict_path = 'data/test_event_new_9_exp71_epoch40_centernet_3_frame_add_no_ball_frame.pkl'
    with open(data_dict_path, 'rb') as f:
        data_dict = pickle.load(f)
    anno_dict = {
        'test_1': 0,
        'test_2': 0,
        'test_3': 0,
        'test_4': 0,
        'test_5': 0,
        'test_6': 0,
        'test_7': 0,
    }
    for img_paths in data_dict.keys():
        game_name = Path(img_paths[0]).parent.name
        anno_dict[game_name] += 1
        
    # pdb.set_trace()
    debug_dir = 'debug/exp10_crop_320_400_resize_182_182_mask_red_ball_resplit_l1_loss_no_weighted_new_augment_epoch=59/relaxed_acc'
    for game_name in sorted(os.listdir(debug_dir)):
        if 'test' in game_name:
            subdir = os.path.join(debug_dir, game_name)
            print(f'sai {game_name}: {len(os.listdir(subdir))} / {anno_dict[game_name]} => {(anno_dict[game_name]-len(os.listdir(subdir))) / anno_dict[game_name]}')



def augment_image():
    img_fp = 'debug/exp4_crop_320_400_resize_182_182_mask_red_ball_resplit_no_weighted_class_epoch=35/relaxed_acc/test_2/228-236/cropped_img_000229.jpg'
    img = cv2.imread(img_fp)

    transforms = A.Compose(
        A.SomeOf([
            A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(p=1, shift_limit=0.1, scale_limit=0.12, rotate_limit=7, border_mode=cv2.BORDER_REPLICATE, value=0),
            A.ColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
            # A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(p=1, alpha=1, sigma=50, alpha_affine=30, border_mode=cv2.BORDER_REPLICATE, value=0),
            A.Perspective(p=1, scale=(0.05, 0.1), keep_size=True, fit_output=True, pad_mode=cv2.BORDER_REPLICATE),
            A.Affine(p=1, scale=(0.8, 1.15), translate_percent=(0, 0.1), rotate=(-7, 7), shear=(-10, 10), mode=cv2.BORDER_REPLICATE)
        ], n=3),
    )

    img = transforms(image=img)['image']
    cv2.imwrite('a.jpg', img)



if __name__ == '__main__':
    get_test_result()
    # augment_image()