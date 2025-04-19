import os
import random
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
import numpy as np

SEED = 42

DEEPFAKES_ORIGINAL_PATH = "data/original_sequences/youtube/c23/videos"
DEEPFAKES_MANIPULATED_PATH = "data/manipulated_sequences/Deepfakes/c23/videos"

F2F_ORIGINAL_PATH = "data/original_sequences/youtube/c23/videos"
F2F_MANIPULATED_PATH = "data/manipulated_sequences/Face2Face/c23/videos"

def deepfakes_title_parser(filename):
    print(filename)
    return filename[:3]
    
def get_video_paths(base_path):
    video_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    return video_files

def generate_video_dataset(
    dataset_type,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    if dataset_type == "deepfakes":
        title_parser=deepfakes_title_parser,    
        original_videos = get_video_paths(DEEPFAKES_ORIGINAL_PATH)
        manipulated_videos = get_video_paths(DEEPFAKES_MANIPULATED_PATH)
    elif dataset_type == "f2f":
        title_parser=deepfakes_title_parser,    
        original_videos = get_video_paths(F2F_ORIGINAL_PATH)
        manipulated_videos = get_video_paths(F2F_MANIPULATED_PATH)
    else:
        raise ValueError("Unsupported dataset type. Choose 'deepfakes' or 'f2f'.")

    title_to_videos = {}
    for video_path in original_videos + manipulated_videos:
        print(dataset_type, video_path)
        base_title = title_parser(os.path.basename(video_path))
        print(base_title)
        
        label = 0 if "original" in video_path else 1
        if base_title not in title_to_videos:
            title_to_videos[base_title] = []
        title_to_videos[base_title].append((video_path, label))
        
    assert all(len(videos) == 2 for videos in title_to_videos.values()), "Should have 2 videos per title (original and manipulated)"

    titles = list(title_to_videos.keys())
    random.seed(SEED)
    random.shuffle(titles)

    total = len(titles)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    def collect_split(titles_subset):
        data = []
        for title in titles_subset:
            data.extend(title_to_videos[title])
        return data

    train_data = collect_split(titles[:train_end])
    val_data = collect_split(titles[train_end:val_end])
    test_data = collect_split(titles[val_end:])

    return train_data, val_data, test_data

class FrameDataset(Dataset):
    def __init__(self, dataset, video_label_pairs, num_frames=5, transform=None, cache_dir="cached_frames"):
        self.dataset = dataset
        self.data = video_label_pairs
        self.num_frames = num_frames
        self.transform = transform
        self.cache_dir = f"{cache_dir}/{dataset}/{num_frames}"
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        video_id = self._video_id(video_path)
        frame_paths = self._get_or_generate_frames(video_path, video_id)

        frames = []
        for frame_path in frame_paths:
            frame = np.load(frame_path)
            frame = torch.from_numpy(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        return frames, torch.tensor(label)

    def _video_id(self, path):
        # Unique ID for caching, based on filename without extension
        return os.path.splitext(os.path.basename(path))[0]

    def _get_or_generate_frames(self, video_path, video_id):
        frame_dir = os.path.join(self.cache_dir, video_id)
        os.makedirs(frame_dir, exist_ok=True)
        frame_paths = [os.path.join(frame_dir, f"frame_{i}.npy") for i in range(self.num_frames)]

        # If all frames exist, return them
        if all(os.path.exists(fp) for fp in frame_paths):
            return frame_paths

        # Otherwise, extract and save them
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            raise ValueError(f"Video {video_path} has fewer frames than requested: {total_frames} < {self.num_frames}")

        indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]
        selected_frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = frame.astype(np.float32) / 255.0  # Normalize
                np.save(frame_paths[indices.index(i)], frame_tensor.transpose(2, 0, 1))  # CHW format
        cap.release()
        return frame_paths
