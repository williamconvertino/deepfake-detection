import os
import json
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.log_dir = os.path.join("logs", model_name)
        self.log_path = self._find_latest_log()
        self.fig_dir = "figures"
        os.makedirs(self.fig_dir, exist_ok=True)

    def _find_latest_log(self):
        all_runs = [d for d in os.listdir(self.log_dir) if os.path.isdir(os.path.join(self.log_dir, d))]
        if not all_runs:
            raise FileNotFoundError(f"No training runs found in logs/{self.model_name}")
        latest_run = sorted(all_runs)[-1]
        return os.path.join(self.log_dir, latest_run, "train_log.json")

    def load_log(self):
        with open(self.log_path, "r") as f:
            return json.load(f)

    def plot(self, save=True, show=False):
        log_data = self.load_log()
        epochs = [entry["epoch"] for entry in log_data]
        train_loss = [entry["loss"] for entry in log_data]
        val_acc = [entry["val_acc"] for entry in log_data]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color=color)
        ax1.plot(epochs, train_loss, color=color, label="Train Loss")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("Val Accuracy", color=color)
        ax2.plot(epochs, val_acc, color=color, linestyle="--", label="Val Accuracy")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Training Curve - {self.model_name}")
        fig.tight_layout()
        plt.grid(True)

        if save:
            save_path = os.path.join(self.fig_dir, f"{self.model_name}_train_curve.png")
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()

        plt.close()