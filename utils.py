import os
import cv2
import torch
import argparse
import numpy as np

    
def collate_train(batch):
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
        model.get_student_type())))

    
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
