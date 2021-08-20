import os
import cv2
import torch
import argparse
import numpy as np
import pickle as pk

from sklearn.model_selection import train_test_split

    
def collate_student(batch):
    anchors, positives, negatives, similarities = zip(*batch)
    videos = anchors + positives + negatives
    num = len(videos)
    max_len = max([s.size(0) for s in videos])
    
    _, r, c = anchors[0].shape
    padded_videos = videos[0].data.new(*(num, max_len, r, c)).fill_(0)
    masks = videos[0].data.new(*(num, max_len)).fill_(0)
    for i, tensor in enumerate(videos):
        length = tensor.size(0)
        padded_videos[i, :length] = tensor
        masks[i, :length] = 1

    similarities = torch.cat(similarities, 0)
    return padded_videos, masks, similarities


def collate_selector(batch):
    queries, targes, similarities, labels = zip(*batch)
    videos = queries + targes
    num = len(videos)
    max_len = max([s.size(0) for s in videos])
    max_reg = max([s.size(1) for s in videos])
    
    padded_videos = videos[0].data.new(*(num, max_len, max_reg, 512)).fill_(0)
    masks = videos[0].data.new(*(num, max_len)).fill_(0)
    for i, tensor in enumerate(videos):
        length = tensor.size(0)
        padded_videos[i, :length] = tensor
        masks[i, :length] = 1

    similarities = torch.cat(similarities, 0)
    labels = torch.cat(labels, 0)
    return padded_videos, masks, similarities, labels


def collate_eval(batch):
    videos, video_ids = zip(*batch)
    num = len(videos)
    max_len = max([s.size(0) for s in videos])
    max_reg = max([s.size(1) for s in videos])
    dims = videos[0].size(2)

    padded_videos = videos[0].data.new(*(num, max_len, max_reg, dims)).fill_(0)
    masks = videos[0].data.new(*(num, max_len)).fill_(0)
    for i, tensor in enumerate(videos):
        length = tensor.size(0)
        padded_videos[i, :length] = tensor
        masks[i, :length] = 1

    return padded_videos, masks, video_ids


def save_model(args, model, optimizer, global_step):
    d = dict()
    d['model'] = model.state_dict()
    d['optimizer'] = optimizer.state_dict()
    d['global_step'] = global_step
    d['args'] = args
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    torch.save(d, os.path.join(args.experiment_path, 'model_{}.pth'.format(
        model.get_network_name())))

    
def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

        
def center_crop(frame, desired_size):
    if frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[top: top+desired_size, left: left+desired_size, :]
    else: 
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[:, top: top+desired_size, left: left+desired_size, :]

    
def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def load_video(video, all_frames=False, fps=1, cc_size=None, rs_size=None):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    fps_div = fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25
    frames = []
    count = 0
    while cap.isOpened():
        ret = cap.grab()
        if int(count % round(fps / fps_div)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rs_size is not None:
                    frame = resize_frame(frame, rs_size)
                frames.append(frame)
            else:
                break
        count += 1
    cap.release()
    frames = np.array(frames)
    if cc_size is not None:
        frames = center_crop(frames, cc_size)
    return frames


def generate_selector_dataset(threshold, val_size=0.03, coarse_student='cg_student', fine_student='fg_att_student'):
    
    with open('data/trainset_similarities_{}_iter2.pk'.format(fine_student), 'rb') as f:
        pickle_file = pk.load(f)
        index = pickle_file['index']
        similarities_fine = pickle_file['pairs']
    with open('data/trainset_similarities_{}_iter2.pk'.format(coarse_student), 'rb') as f:
        similarities_coarse = pk.load(f)['pairs']
    
    X, y = [], []
    for query, pair_pools in similarities_fine.items():
        for pos in pair_pools['positives']:
            sim_fine = similarities_fine[query]['positives'][pos] / 2. + 0.5
            sim_coarse = similarities_coarse[query]['positives'][pos]
            
            x = [index[query], index[pos], sim_coarse]
            X.append(np.array(x))
            
            l = 1 if np.abs(sim_fine - sim_coarse) > threshold else 0
            y.append(l)
        for neg in pair_pools['negatives']:
            sim_fine = similarities_fine[query]['negatives'][neg] / 2. + 0.5
            sim_coarse = similarities_coarse[query]['negatives'][neg]
            
            x = [index[query], index[neg], sim_coarse]
            X.append(np.array(x))
            
            l = 1 if np.abs(sim_fine - sim_coarse) > threshold else 0
            y.append(l)

    X = np.array(X)
    y = np.array(y, dtype=np.float32)
    return train_test_split(X, y, test_size=val_size, random_state=42)
