import random
import os
from pathlib import Path

import torch
from torchvision.transforms import v2

def label_to_emotion(label):
    emotions = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }
    return emotions.get(label, "Unknown")


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, model_path: str | Path) -> None:
    model.load_state_dict(torch.load("models/" + model_path, weights_only=True))
    model.eval()
    return model


def generate_experiment_name():
    model_dir = Path("models")

    if not model_dir.exists():
        model_dir.mkdir()

    adjectives = ["extravagant", "luscious", "purple", "magnificent", "perfect", "charming"]

    fruits = ["olive", "grape", "banana", "apple", "grapefruit", "orange"]

    return f"{len(list(model_dir.iterdir()))}-{random.choice(adjectives)}-{random.choice(fruits)}"

def preprocess_frame(frame):
    resizer = v2.Resize((48, 48))
    frame = resizer(frame.expand(1,1, -1, -1)).float().cuda() / 255.0
    
    return frame
