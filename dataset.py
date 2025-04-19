import os
import random
import cv2
import torch
from torch.utils.data import Dataset

SEED = 42

DEEPFAKES_ORIGINAL_PATH = "data/original_sequences/actors/c23/videos"
DEEPFAKES_MANIPULATED_PATH = "data/manipulated_sequences/DeepFakeDetection/c23/videos"

def deepfakes_title_parser(filename):
    return filename.split("_")[0]

def get_video_paths(base_path):
    video_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    return video_files

def generate_video_dataset(
    original_path=DEEPFAKES_ORIGINAL_PATH,
    manipulated_path=DEEPFAKES_MANIPULATED_PATH,
    title_parser=deepfakes_title_parser,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    original_videos = get_video_paths(original_path)
    manipulated_videos = get_video_paths(manipulated_path)

    title_to_videos = {}
    for video_path in original_videos + manipulated_videos:
        filename = os.path.basename(video_path)
        base_title = title_parser(filename)
        label = 0 if "original" in video_path else 1
        if base_title not in title_to_videos:
            title_to_videos[base_title] = []
        title_to_videos[base_title].append((video_path, label))

    print([len(videos) for videos in title_to_videos.values()])

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
    def __init__(self, video_label_pairs, num_frames=5, transform=None):
        self.data = video_label_pairs
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.extract_frames(video_path, self.num_frames)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, torch.tensor(label)

    def extract_frames(self, path, num_frames):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            raise ValueError(f"Video {path} has fewer frames than requested: {total_frames} < {num_frames}")
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        selected_frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                selected_frames.append(frame_tensor)
        cap.release()
        return selected_frames
