import os
from pathlib import Path
import cv2
import numpy as np
import json
import pickle
import pdb
import albumentations as A





def get_experiment_dir(root_dir, description=None):
    os.makedirs(root_dir, exist_ok=True)
    exp_nums = [int(subdir[3:]) if '_' not in subdir else int(subdir.split('_')[0][3:]) for subdir in os.listdir(root_dir)]
    max_exp_num = max(exp_nums) if len(exp_nums) > 0 else 0
    exp_name = f'exp{max_exp_num+1}' if description is None else f'exp{max_exp_num+1}_{description}'
    return os.path.join(root_dir, exp_name)



def create_paths2pos(data_dir, n_input_frames, mode='train'):
    path2pos = {}
    for jp in Path(data_dir).rglob('ball_markup.json'):
        print(f'processing {jp}')

        img_dir = str(jp.parent).replace('/annotations/', '/images/')
        data = json.load(open(jp))
        for fr in sorted(data.keys()):
            if data[fr]['x'] <= 0 or data[fr]['y'] <= 0:
                continue
            pos = data[fr]['x'] / 1920, data[fr]['y'] / 1080
            fr = int(fr)
            if fr < n_input_frames:
                continue
            ls_fr = [fr-i for i in range(n_input_frames-1, -1, -1)]  # từ nhỏ đến lớn
            ls_pos = []
            for el in ls_fr:
                if str(el) in data:
                    ls_pos.append((data[str(el)]['x']/1920, data[str(el)]['y']/1080))
                else:
                    ls_pos.append((-100, -100))

            ls_img_fp = [os.path.join(img_dir, 'img_' + '{:06d}'.format(fr) + '.jpg') for fr in ls_fr]
            ls_img_fp = [fp for fp in ls_img_fp if os.path.exists(fp)]
            if len(ls_img_fp) == n_input_frames:   # có đủ ảnh
                path2pos[tuple(ls_img_fp)] = ls_pos
    
    if mode == 'train':
        keys = list(path2pos.keys())
        np.random.seed(42)
        np.random.shuffle(keys)

        train_keys = keys[:int(0.8*len(keys))]
        val_keys = keys[int(0.8*len(keys)):]

        train_dict = {k: path2pos[k] for k in train_keys}
        val_dict = {k: path2pos[k] for k in val_keys}

        train_bin = pickle.dumps(train_dict)
        with open(f'data/gpu2_train_dict_{n_input_frames}.pkl', 'wb') as f:
            f.write(train_bin)
        
        val_bin = pickle.dumps(val_dict)
        with open(f'data/gpu2_val_dict_{n_input_frames}.pkl', 'wb') as f:
            f.write(val_bin)

    elif mode == 'test':
        test_bin = pickle.dumps(path2pos)
        with open(f'data/gpu2_test_dict_{n_input_frames}.pkl', 'wb') as f:
            f.write(test_bin)
            
    print('Done')
    return path2pos



# Define the human_readable_size function
def human_readable_size(t):
    # Get the size of A in bytes
    size_bytes = t.numel() * t.element_size()
    
    # Define the units and their multipliers
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    multipliers = [1024 ** i for i in range(len(units))]

    # Find the appropriate unit and multiplier for the input size
    for unit, multiplier in zip(units[::-1], multipliers[::-1]):
        if size_bytes >= multiplier:
            size = size_bytes / multiplier
            return f"{size:.2f} {unit}"

    # If the input size is smaller than 1 byte, return it as-is
    return f"{size_bytes} B"


def smooth_event_labelling(event_class, smooth_idx, event_frameidx):
    target_events = np.zeros((2,))
    if event_class != 2:    # 2 = empty_event
        n = smooth_idx - event_frameidx
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.
    return tuple(target_events)


def get_events_infor(
        root_dir,
        game_list, 
        n_input_frames=9,
        mode='train',
        smooth_labeling=True,
        event_dict={
            'bounce': 0,
            'net': 1,
            'empty_event': 2
        }
    ):
    # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((n_input_frames - 1) / 2)

    annos_dir = os.path.join(root_dir, mode, 'annotations')
    images_dir = os.path.join(root_dir, mode, 'images')
    events_infor = {}
    events_labels = []
    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')
        events_annos_path = os.path.join(annos_dir, game_name, 'events_markup.json')
        # Load ball annotations
        json_ball = open(ball_annos_path)
        ball_annos = json.load(json_ball)
        # Load events annotations
        json_events = open(events_annos_path)
        events_annos = json.load(json_events)

        for event_frameidx, event_name in events_annos.items():
            event_frameidx = int(event_frameidx)
            smooth_frame_indices = [event_frameidx]  # By default
            if (event_name != 'empty_event') and smooth_labeling:
                smooth_frame_indices = [idx for idx in range(event_frameidx - num_frames_from_event,
                                                             event_frameidx + num_frames_from_event + 1)]  # 9 indices

            for smooth_idx in smooth_frame_indices:
                sub_smooth_frame_indices = [idx for idx in range(smooth_idx - num_frames_from_event,
                                                                 smooth_idx + num_frames_from_event + 1)]
                img_path_list = []
                for sub_smooth_idx in sub_smooth_frame_indices:
                    img_path = os.path.join(images_dir, game_name, 'img_{:06d}.jpg'.format(sub_smooth_idx))
                    img_path_list.append(img_path)
                last_f_idx = smooth_idx + num_frames_from_event
                # Get ball position for the last frame in the sequence
                if str(last_f_idx) not in ball_annos.keys():
                    print('{}, smooth_idx: {} - no ball position for the frame idx {}'.format(game_name, smooth_idx, last_f_idx))
                    continue

                ls_ball_pos = [ball_annos[str(f_idx)] if str(f_idx) in ball_annos else {'x': -100, 'y': -100} for f_idx in sub_smooth_frame_indices]
                ls_ball_pos = [(pos['x']/1920, pos['y']/1080) for pos in ls_ball_pos]
                ball_position_xy = ls_ball_pos[-1]

                # Ignore the event without ball information
                if (ball_position_xy[0] < 0) or (ball_position_xy[1] < 0):
                    continue

                # Get segmentation path for the last frame in the sequence
                seg_path = os.path.join(annos_dir, game_name, 'segmentation_masks', '{}.png'.format(last_f_idx))
                if not os.path.isfile(seg_path):
                    print("smooth_idx: {} - The segmentation path {} is invalid".format(smooth_idx, seg_path))
                    continue

                event_class = event_dict[event_name]
                target_events = smooth_event_labelling(event_class, smooth_idx, event_frameidx)
                events_infor[tuple(img_path_list)] = [ls_ball_pos, target_events, event_class, seg_path]
                # Re-label if the event is neither bounce nor net hit
                if (target_events[0] == 0) and (target_events[1] == 0):
                    event_class = 2
                events_labels.append(event_class)

    return events_infor, events_labels


import shutil
def gen_data_for_ball_detection(pkl_fp, save_dir):
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with open(pkl_fp, 'rb') as f:
        data = pickle.load(f)

    ball_radius = 20
    cnt = 0
    for img_paths, labels in data.items():
        img_fp = Path(img_paths[-1])
        pos = labels[-1]
        game_name = img_fp.parent.name
        # pdb.set_trace()
        x_center, y_center = pos
        w = ball_radius / 1920
        h = ball_radius / 1080
        yolo_anno = f'0 {x_center} {y_center} {w} {h}'
        # save image
        out_img_fp = os.path.join(images_dir, f'{game_name}_{img_fp.stem}.jpg')
        shutil.copy(str(img_fp), out_img_fp)

        # save annotation
        out_anno_fp = os.path.join(labels_dir, f'{game_name}_{img_fp.stem}.txt')
        with open(out_anno_fp, 'w') as f:
            f.write(yolo_anno)

        cnt += 1
        print(f'Done {cnt} images')


def gen_data_for_event_cls(ev_data_fp, split):
    with open(ev_data_fp, 'rb') as f:
        ev_data = pickle.load(f)
    all_img_dict = {}
    for res_split in ['train', 'val', 'test']:
        result_fp = f'results/exp38_centernet_ason_v8n_fixed_mask_ball/{res_split}/result.json'
        result_data = json.load(open(result_fp))
        all_img_dict.update(result_data['img_dict'])
    # pdb.set_trace()
    all_img_paths = sorted(list(all_img_dict.keys()))
    final_dict = {}
    for img_paths, labels in ev_data.items():
        cnt = 0
        ls_pos = []
        for fp in img_paths:
            if fp in all_img_paths:
                pred = (all_img_dict[fp]['pred'][0]/512, all_img_dict[fp]['pred'][1]/512)
                ls_pos.append(pred)
            else:
                ls_pos.append((-1, -1))
                cnt += 1
        if cnt <= 2:
            final_dict[tuple(img_paths)] = (ls_pos, labels[1])
    
    bin = pickle.dumps(final_dict)
    with open(f'data/{split}_event_new_9.pkl', 'wb') as f:
        f.write(bin)



def crop_img_for_event_cls(ev_data_fp, out_dir, split, crop_size=128):
    with open(ev_data_fp, 'rb') as f:
        ev_data = pickle.load(f)
    new_ev_data = {}
    for img_paths, (ls_pos, ev_target) in ev_data.items():
        new_img_paths = []
        for i, img_fp in enumerate(img_paths):
            orig_img = cv2.imread(img_fp)
            img_fp = Path(img_fp)
            orig_h, orig_w = orig_img.shape[:2]
            pos = ls_pos[i]

            if pos[0] <= 0 or pos[1] <= 0:
                cropped_img = np.zeros(shape=(crop_size, crop_size, 3), dtype=np.uint8)
            else:
                orig_pos = (int(pos[0]*orig_w), int(pos[1]*orig_h))
                r = crop_size//2
                xmin, ymin = orig_pos[0] - r, orig_pos[1] - r
                xmax, ymax = orig_pos[0] + r, orig_pos[1] + r
                cropped_img = orig_img[max(0, ymin):max(0, ymax), max(0, xmin):max(0, xmax)]
                if cropped_img.shape != (crop_size, crop_size, 3):
                    pad_x = (max(0, -xmin), max(0, xmax-orig_w))
                    pad_y = (max(0, -ymin), max(0, ymax-orig_h))
                    pad_c = (0, 0)
                    cropped_img = np.pad(cropped_img, [pad_y, pad_x, pad_c], mode='constant')

            out_fp = os.path.join(out_dir, img_fp.parent.name, img_fp.name)
            cnt = 1
            while os.path.exists(out_fp):
                out_fp = Path(out_fp)
                out_fp = out_fp.parent / (out_fp.stem + f'_{cnt}' + out_fp.suffix)
                cnt += 1
            out_fp = str(out_fp)

            new_img_paths.append(out_fp)
            os.makedirs(Path(out_fp).parent, exist_ok=True)
            cv2.imwrite(out_fp, cropped_img)
            print(f'Done {out_fp}')
        new_ev_data[tuple(new_img_paths)] = ev_data[img_paths]

    bin = pickle.dumps(new_ev_data)
    with open(f'data/{split}_event_cropped_9_128_128.pkl', 'wb') as f:
        f.write(bin)


def crop_img_for_event_cls_2(ev_data_fp, out_dir, split, crop_size=(320, 128)):
    with open(ev_data_fp, 'rb') as f:
        ev_data = pickle.load(f)
    new_ev_data = {}
    for img_paths, (ls_pos, ev_target) in ev_data.items():
        new_img_paths = []
        # cx_min = np.clip(min([pos[0] for pos in ls_pos if pos[0] > 0]), 0, 1)
        # cx_max = np.clip(max([pos[0] for pos in ls_pos if pos[0] > 0]), 0, 1)
        # cy_min = np.clip(min([pos[1] for pos in ls_pos if pos[1] > 0]), 0, 1)
        # cy_max = np.clip(max([pos[1] for pos in ls_pos if pos[1] > 0]), 0, 1)

        # cx_min *= 1920
        # cx_max *= 1920
        # cy_min *= 1080
        # cy_max *= 1080

        # cx_min -= 20
        # cx_max += 20
        # cy_min -= 20
        # cy_max += 20

        # cx_min = np.clip(cx_min, 0, 1920)
        # cx_max = np.clip(cx_max, 0, 1920)
        # cy_min = np.clip(cy_min, 0, 1080)
        # cy_max = np.clip(cy_max, 0, 1080)


        mean_cx = np.mean([pos[0] for pos in ls_pos if pos[0] > 0])
        mean_cy = np.mean([pos[1] for pos in ls_pos if pos[1] > 0])
        # mean_cx = ls_pos[4][0]
        
        mean_cx = int(mean_cx*1920)
        mean_cy = int(mean_cy*1080)
        xmin = max(0, mean_cx - crop_size[0]//2)
        xmax = min(mean_cx + crop_size[0]//2, 1920)
        ymin = max(0, mean_cy - crop_size[1]//2)
        ymax = min(mean_cy + crop_size[1]//2, 1080)

        # xmin, xmax, ymin, ymax = 650, 1000, 500, 700



        for i, img_fp in enumerate(img_paths):
            orig_img = cv2.imread(img_fp)
            img_fp = Path(img_fp)
            # cropped_img = orig_img[int(cy_min):int(cy_max), int(cx_min):int(cx_max)]
            # print(xmin, xmax, ymin, ymax)
            cropped_img = orig_img[ymin:ymax, xmin:xmax]
            # print(cropped_img.shape)
            try:
                cropped_img = cv2.resize(cropped_img, crop_size)
            except:
                pdb.set_trace()
            out_fp = os.path.join(out_dir, img_fp.parent.name, img_fp.name)
            cnt = 1
            while os.path.exists(out_fp):
                out_fp = Path(out_fp)
                out_fp = out_fp.parent / (out_fp.stem + f'_{cnt}' + out_fp.suffix)
                cnt += 1
            out_fp = str(out_fp)
            new_img_paths.append(out_fp)
            os.makedirs(Path(out_fp).parent, exist_ok=True)
            cv2.imwrite(out_fp, cropped_img)
            print(f'Done {out_fp}')
        new_ev_data[tuple(new_img_paths)] = (ls_pos, ev_target)

    bin = pickle.dumps(new_ev_data)
    with open(f'data/{split}_event_cropped_9_320_128.pkl', 'wb') as f:
        f.write(bin)


if __name__ == '__main__':
    np.random.seed(42)

    for split in ['test', 'val', 'train']:
        if split != 'train':
            continue
        ev_data_fp = f'data/{split}_event_new_9.pkl'
        out_dir = f'cropped_data_320_128/{split}'
        crop_img_for_event_cls_2(ev_data_fp, out_dir, split)

    # with open('data/test_event_cropped_9_128_128.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # items = list(data.items())
    # print(items[10])
    # pdb.set_trace()

    # for split in ['train', 'val', 'test']:
    #     ev_data_fp = f'data/gpu2_event_{split}_dict_9.pkl'
    #     gen_data_for_event_cls(ev_data_fp, split)

    # bounce, net, empty = 0, 0, 0
    # for split in ['train', 'val', 'test']:
    #     with open(f'data/{split}_event_new_9.pkl', 'rb') as f:
    #         obj = pickle.load(f)
    #     items = list(obj.items())
    #     ls_ev = [el[1][1] for el in items]
    #     for ev in ls_ev:
    #         if ev[0] != 0:
    #             bounce += 1
    #         elif ev[1] != 0:
    #             net += 1
    #         elif ev[0] == 0 and ev[1] == 0:
    #             empty += 1
        
    # print('bounce, net, empty: ', bounce, net, empty)
    # pdb.set_trace()


    # pkl_fp = 'data/gpu2_val_dict_5.pkl'
    # save_dir = '/data2/tungtx2/datn/yolov8/ball_detection_data/val'
    # gen_data_for_ball_detection(pkl_fp, save_dir)


    # data_dir = '/data2/tungtx2/datn/ttnet/dataset/test'
    # n_input_frames = 5
    # path2pos = create_paths2pos(data_dir, n_input_frames, mode='test')

    # n_input_frames = 9
    # mode = 'test'
    # ev_info, ev_labels = get_events_infor(
    #     root_dir='/data2/tungtx2/datn/ttnet/dataset/',
    #     # game_list=['game_1', 'game_2', 'game_3', 'game_4', 'game_5'],
    #     game_list=['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7'],
    #     n_input_frames=n_input_frames,
    #     mode=mode,
    #     smooth_labeling=True,
    #     event_dict={
    #         'bounce': 0,
    #         'net': 1,
    #         'empty_event': 2
    #     }
    # )

    # if mode == 'train':
    #     keys = list(ev_info.keys())
    #     np.random.shuffle(keys)

    #     train_keys = keys[:int(0.85*len(keys))]
    #     val_keys = keys[int(0.85*len(keys)):]

    #     ev_train = {k: ev_info[k] for k in train_keys}
    #     ev_val = {k: ev_info[k] for k in val_keys}

    #     train_bin = pickle.dumps(ev_train)
    #     with open(f'data/gpu2_event_train_dict_{n_input_frames}.pkl', 'wb') as f:
    #         f.write(train_bin)

    #     val_bin = pickle.dumps(ev_val)
    #     with open(f'data/gpu2_event_val_dict_{n_input_frames}.pkl', 'wb') as f:
    #         f.write(val_bin)
    # else:
    #     test_bin = pickle.dumps(ev_info)
    #     with open(f'data/gpu2_event_test_dict_{n_input_frames}.pkl', 'wb') as f:
    #         f.write(test_bin)




    # with open('data/gpu2_event_val_dict_9.pkl', 'rb') as f:
    #     obj = pickle.load(f)
    # pdb.set_trace()