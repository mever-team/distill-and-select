import h5py
import torch
import random
import numpy as np
import pickle as pk

from torch.utils.data import Dataset

        
class DatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, feature_file, videos, min_len=4):
        self.feature_file = h5py.File(feature_file, "r")
        self.videos = videos
        self.min_len = min_len

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        try:
            video_id = self.videos[idx]
            features = self.feature_file[video_id][:]
            while features.shape[0] < self.min_len:
                features = np.concatenate([features, features], axis=0)
            if features.ndim == 2: 
                features = np.expand_dims(features, 1)
            features = torch.from_numpy(features.astype(np.float32))
            return features, video_id
        except Exception as e:
            return torch.zeros((1, 9, 512)), ''


class StudentPairGenerator(Dataset):

    def __init__(self, args):
        super(StudentPairGenerator, self).__init__()
        ground_truths = pk.load(open('data/trainset_similarities_{}.pk'.format(args.teacher), 'rb'))
        self.index = ground_truths['index']
        self.pairs = ground_truths['pairs']
        self.feature_file = h5py.File(args.trainset_hdf5, "r")
        self.augmentation = args.augmentation
        self.selected_pairs = []
        self.normalize = args.student_type == 'coarse-grained'

        self.videos = set(self.feature_file.keys())
        self.video_set = [i for i, v in enumerate(self.index) if v in self.videos]
        np.random.shuffle(self.video_set)
        self.video_set = set(self.video_set[:int(len(self.index)*args.trainset_percentage/100)])

    def next_epoch(self, seed=42):
        self.selected_pairs = self.select_pairs(seed=seed)

    def select_pairs(self, seed=42):
        np.random.seed(seed)
        selected_pairs = []
        for q, t in self.pairs.items():
            pos = [v for v in list(t['positives'].keys()) if v in self.video_set]
            neg = [v for v in list(t['negatives'].keys()) if v in self.video_set]
            if q in self.video_set and pos and neg:
                p = random.choice(pos)
                n = random.choice(neg)
                sim_p = t['positives'][p]
                sim_n = t['negatives'][n]
                if self.normalize:
                    sim_p = sim_p / 2. + 0.5
                    sim_n = sim_n / 2. + 0.5
                selected_pairs.append([q, p, n, float(sim_p), float(sim_n)])
        return selected_pairs
                        
    def load_video(self, video, augmentation=False):
        video_tensor = self.feature_file[str(self.index[video])][:]
        if augmentation:
            video_tensor = self.augment(video_tensor)
        return torch.from_numpy(video_tensor.astype(np.float32))

    def augment(self, video):
        if video.shape[0] > 8:
            rnd = np.random.uniform()
            if rnd < 0.2:
                N, T, D = video.shape
                window_size = np.random.randint(4, 16)
                offset = N % window_size
                if offset:
                    video = np.concatenate([video, video], 0)[:N + (window_size - offset)]
                video = video.reshape(-1, window_size, T, D)
                if rnd < 0.1:
                    mask = np.random.rand(video.shape[0]) > 0.3
                    if np.sum(mask):
                        video = video[mask]
                else:
                    np.random.shuffle(video)
                video = np.reshape(video, (-1, T, D))
            elif rnd < 0.3:
                video = video[::2]
            elif rnd < 0.4:
                if video.shape[0] < 150:
                    idx = np.insert(np.arange(len(video)), np.arange(len(video)), np.arange(len(video)))
                    video = video[idx]
            elif rnd < 0.5:
                video = video[::-1]
        return video[:300]

    def __len__(self):
        return len(self.selected_pairs)

    def __getitem__(self, idx):
        pairs = self.selected_pairs[idx]
        anchor = self.load_video(pairs[0])
        positive = self.load_video(pairs[1], augmentation=self.augmentation)
        negative = self.load_video(pairs[2], augmentation=self.augmentation)
        simimarities = torch.tensor(pairs[3:]).unsqueeze(0)
        return anchor, positive, negative, simimarities

    
class SelectorPairGenerator(Dataset):

    def __init__(self, video_pairs, labels, args):
        super(SelectorPairGenerator, self).__init__()
        self.video_pairs = video_pairs
        self.labels = labels
        self.feature_file = h5py.File(args.trainset_hdf5, "r")
        self.pairs = self.sample_pairs()
    
    def next_epoch(self, size=None):
        self.pairs = self.sample_pairs(size)
        
    def sample_pairs(self, size=None):
        unique, counts = np.unique(self.labels, return_counts=True)
        if size is None or size > np.min(counts):
            size = np.min(counts)

        ids = []
        for u in unique:
            ids.extend(np.random.choice(np.where(self.labels == u)[0], size, replace=False).tolist())
        return ids
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx = self.pairs[idx]
        pair = self.video_pairs[idx]
        query = torch.from_numpy(self.feature_file[pair[0]][:])
        target = torch.from_numpy(self.feature_file[pair[1]][:])
        similarity = torch.tensor([float(pair[2])])
        label = torch.tensor(self.labels[idx]).unsqueeze(0)
        return query, target, similarity, label
