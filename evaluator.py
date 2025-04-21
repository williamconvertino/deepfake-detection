import torch
from torch.utils.data import DataLoader
from dataset import generate_video_dataset, FrameDataset
import torchvision.transforms as T
import re
import os
import json
from datetime import datetime
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataset, num_frames=5, batch_size=16, device=None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset
        self.num_frames = num_frames
        self.batch_size = batch_size
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def prepare_data(self):
        _, val_loader, test_data = generate_video_dataset(dataset_type=self.dataset)
        self.test_loader = DataLoader(FrameDataset(self.dataset, test_data, self.num_frames, transform=self.transform), batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(FrameDataset(self.dataset, val_loader, self.num_frames, transform=self.transform), batch_size=self.batch_size, shuffle=False)

    def aggregate_logits(self, logits, aggregation="average"):
        b, t, c = logits.size()
        if aggregation == "average":
            return logits.mean(dim=1)
        vote_match = re.match(r"(\d+)_vote", aggregation)
        if vote_match:
            threshold = int(vote_match.group(1))
            preds = logits.argmax(dim=2)
            counts = torch.stack([(preds == i).sum(dim=1) for i in range(c)], dim=1)
            return counts
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    def evaluate(self, aggregation="average", split="test"):
        print(f"Evaluating with aggregation method: {aggregation}")

        self.prepare_data()

        total_videos = 0
        correct_videos = 0
        total_frames = 0
        correct_frames = 0

        pos_vote_counts_pos_videos = []
        neg_vote_counts_pos_videos = []
        pos_vote_counts_neg_videos = []
        neg_vote_counts_neg_videos = []

        if split == "val":
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        with torch.no_grad():
            for frames, labels in tqdm(data_loader, desc="Evaluating", unit="batch"):
                b, t, c, h, w = frames.size()
                frames = frames.view(b * t, c, h, w).to(self.device)
                labels = labels.to(self.device)

                logits = self.model(frames)
                logits = logits.view(b, t, -1)

                frame_preds = logits.argmax(dim=2)
                correct_frames += (frame_preds == labels.unsqueeze(1)).sum().item()
                total_frames += b * t

                if aggregation == "average":
                    agg_logits = self.aggregate_logits(logits, aggregation=aggregation)
                    final_preds = agg_logits.argmax(dim=1)
                elif "_vote" in aggregation:
                    counts = self.aggregate_logits(logits, aggregation=aggregation)
                    final_preds = counts.argmax(dim=1)
                    for i in range(b):
                        if labels[i] == 1:
                            pos_vote_counts_pos_videos.append(counts[i, 1].item())
                            neg_vote_counts_pos_videos.append(counts[i, 0].item())
                        else:
                            pos_vote_counts_neg_videos.append(counts[i, 1].item())
                            neg_vote_counts_neg_videos.append(counts[i, 0].item())
                else:
                    raise ValueError("Unsupported aggregation type")

                correct_videos += (final_preds == labels).sum().item()
                total_videos += b

        frame_acc = correct_frames / total_frames
        video_acc = correct_videos / total_videos

        def compute_vote_stats(pos_votes, neg_votes):
            n = len(pos_votes)
            if n == 0:
                return {"avg_pos_votes": None, "avg_pos_percent": None,
                        "avg_neg_votes": None, "avg_neg_percent": None}
            return {
                "avg_pos_votes": sum(pos_votes) / n,
                "avg_pos_percent": sum(pos_votes) / (n * self.num_frames),
                "avg_neg_votes": sum(neg_votes) / n,
                "avg_neg_percent": sum(neg_votes) / (n * self.num_frames)
            }

        pos_video_stats = compute_vote_stats(pos_vote_counts_pos_videos, neg_vote_counts_pos_videos)
        neg_video_stats = compute_vote_stats(pos_vote_counts_neg_videos, neg_vote_counts_neg_videos)

        results = {
            "aggregation": aggregation,
            "video_accuracy": video_acc,
            "frame_accuracy": frame_acc,
            "video_stats": {
                "positive_videos": pos_video_stats,
                "negative_videos": neg_video_stats
            },
            "meta": {
                "total_videos": total_videos,
                "total_frames": total_frames,
                "correct_videos": correct_videos,
                "correct_frames": correct_frames,
                "num_frames_per_video": self.num_frames
            }
        }

        print(json.dumps(results, indent=2))

        log_dir = os.path.join("logs", self.model.name)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"test_results_{aggregation}_{timestamp}.json")

        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved evaluation results to {log_path}")
        
        return results
    
    def search_evaluate(self):
        
        aggregations = ["average", "40_vote", "50_vote", "60_vote", "70_vote"]
        best_results = None
        
        for agg in aggregations:
            print(f"Evaluating with aggregation method: {agg}")
            results = self.evaluate(aggregation=agg, split="val")
            
            if best_results is None or results["video_accuracy"] > best_results["video_accuracy"]:
                best_results = results
                
        print(f"Best aggregation method: {best_results['aggregation']}")
        print(best_results)