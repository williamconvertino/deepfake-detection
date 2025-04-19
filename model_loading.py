import os
import yaml
import torch
import torchvision.models as models
from models.vit import ViT

CHECKPOINT_DIR = "checkpoints"

def load_config(config_name):
    config_path = os.path.join("config", f"{config_name}.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model_instance(config):
    model_type = config.get("model_type", "resnet18")
    
    if model_type == "resnet18":
        model = models.resnet18(pretrained=config.get("pretrained", True))
        model.fc = torch.nn.Linear(model.fc.in_features, 2) # 2 classes (original and manipulated)
    elif model_type == "vit":
        model = ViT()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

# Note that checkpoint should either be "best" or "epoch_<number>" (or None if you want to train from scratch)
def load_model(config_name, checkpoint=None):
    config = load_config(config_name)
    
    model = get_model_instance(config)

    model.name = config_name
    
    checkpoint_path = None
    if checkpoint:
        if checkpoint == "best":
            checkpoint_path = os.path.join(CHECKPOINT_DIR, config_name, "best.pt")
        elif checkpoint.startswith("epoch_"):
            epoch_num = checkpoint.split("_")[1]
            checkpoint_path = os.path.join(CHECKPOINT_DIR, config_name, f"{epoch_num}.pt")
        else:
            raise ValueError(f"Invalid checkpoint format: {checkpoint}")

        if os.path.exists(checkpoint_path):
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint_data["model_state_dict"])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


    return model
