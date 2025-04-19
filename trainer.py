import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datetime import datetime
import json
from dataset import generate_video_dataset, FrameDataset

class Trainer:
    def __init__(self, model, num_frames=5, batch_size=8, epochs=10, lr=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model.name
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

        self.best_val_acc = 0
        self.checkpoint_dir = os.path.join("checkpoints", self.model_name)
        self.log_dir = os.path.join("logs", self.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, "train_log.json")
        self.history = []

    def prepare_data(self):
        train_data, val_data, test_data = generate_video_dataset()
        self.train_loader = DataLoader(FrameDataset(train_data, self.num_frames, transform=self.transform), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(FrameDataset(val_data, self.num_frames, transform=self.transform), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(FrameDataset(test_data, self.num_frames, transform=self.transform), batch_size=self.batch_size, shuffle=False)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for frames, labels in self.train_loader:
            b, t, c, h, w = frames.size()
            frames = frames.view(b * t, c, h, w).to(self.device)
            labels = labels.to(self.device)

            logits = self.model(frames)
            logits = logits.view(b, t, -1).mean(dim=1)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        return total_loss / len(self.train_loader), acc

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for frames, labels in self.val_loader:
                b, t, c, h, w = frames.size()
                frames = frames.view(b * t, c, h, w).to(self.device)
                labels = labels.to(self.device)

                logits = self.model(frames)
                logits = logits.view(b, t, -1).mean(dim=1)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def save_checkpoint(self, epoch, train_loss, val_acc, is_best=False):
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "train_loss": train_loss,
            "val_acc": val_acc
        }

        path = os.path.join(self.checkpoint_dir, f"{epoch}.pt")
        torch.save(checkpoint_data, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint_data, best_path)

    def log_epoch(self, epoch, loss, train_acc, val_acc):
        entry = {
            "epoch": epoch,
            "loss": loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        }
        self.history.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self):
        print(f"Training {self.model_name} on {self.device}")
        self.prepare_data()
        for epoch in range(1, self.epochs + 1):
            loss, train_acc = self.train_epoch()
            val_acc = self.validate()
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            self.log_epoch(epoch, loss, train_acc, val_acc)
            self.save_checkpoint(epoch, train_loss=loss, val_acc=val_acc, is_best=val_acc > self.best_val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
