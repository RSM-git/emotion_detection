import torch
import torch.nn as nn

from emotion_detection.model import EmotionNet
from emotion_detection.dataset import get_train_and_val_dataloader, get_test_dataloader
from emotion_detection.metrics import Accuracy, Recall, Precision
from emotion_detection import utils

from tqdm import tqdm

train_dataloader, val_dataloader = get_train_and_val_dataloader()
test_dataloader = get_test_dataloader()

model = EmotionNet()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

epochs: int = 100

exp_name = utils.generate_experiment_name()

metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(num_classes=7, device=torch.device("cuda")),
        "recall": Recall(num_classes=7, device=torch.device("cuda")),
        }

best_val_accuracy = 0.0

class Trainer:
    def __init__(self):
        pass

    def train(self, model, train_dataloader, val_dataloader):
        pass

for epoch in tqdm(range(epochs)):
    
    model.train()

    for metric_name, metric in metrics.items():
        metric.reset()

    for batch in train_dataloader:
        img = batch["data"].cuda()
        label = batch["label"].cuda()
        
        out = model(img)

        loss = criterion(out, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for metric_name, metric in metrics.items():
            metric.update(out, label)
        
    print(f"Epoch {epoch+1}/{epochs} Train Metrics:")
    for metric_name, metric in metrics.items():
        print(f"{metric_name}: {metric.compute()}")

    # Validation
    model.eval()
    for metric_name, metric in metrics.items():
        metric.reset()
    
    with torch.no_grad():
        for batch in val_dataloader:
            img = batch["data"].cuda()
            label = batch["label"].cuda()
            
            out = model(img)
            
            for metric_name, metric in metrics.items():
                metric.update(out, label)
        
    print(f"Epoch {epoch+1}/{epochs} Validation Metrics:")

    for metric_name, metric in metrics.items():
        print(f"{metric_name}: {metric.compute()}")

    if metrics["accuracy"].value > best_val_accuracy:
        best_val_accuracy = metrics["accuracy"].value
        utils.save_model(model, f"models/{exp_name}.pt")
