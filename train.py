import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emotion_detection.metrics import Accuracy, Recall, Precision
from emotion_detection import utils

from tqdm import tqdm

class Trainer:
    def __init__(self, model: nn.Module, **kwargs):
        self.epochs = kwargs["epochs"]
        self.batch_size = kwargs["batch_size"]
        self.weight_decay = kwargs["weight_decay"]
        self.lr = kwargs["learning_rate"]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metrics = {
                "accuracy": Accuracy(),
                "precision": Precision(num_classes=7, device=torch.device("cuda")),
                "recall": Recall(num_classes=7, device=torch.device("cuda")),
        }

    def _update_metrics(self, out, label):
        for metric_name, metric in self.metrics.items():
            metric.update(out, label)

    def _reset_metrics(self):
        for metric_name, metric in self.metrics.items():
            metric.reset()

    def _shared_step(self, batch):
        img = batch["data"].to(self.device)
        label = batch["label"].to(self.device)

        return self.model(img), label


    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        exp_name: str = utils.generate_experiment_name()
        best_val_accuracy: float = 0.0

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
        )

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self._reset_metrics()
            
            for batch in train_dataloader:
                out, label = self._shared_step(batch)

                loss = criterion(out, label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self._update_metrics(out, label)

            print(f"Epoch {epoch+1}/{epochs} Train Metrics:")
            for metric_name, metric in self.metrics.items():
                print(f"{metric_name}: {metric.compute()}")
            
            self.model.eval()
            self._reset_metrics()

            with torch.no_grad():
                for batch in val_dataloader:
                    out, label = self._shared_step(batch)

                    self._update_metrics(out, label)


            print(f"Epoch {epoch+1}/{epochs} Validation Metrics:")

            for metric_name, metric in metrics.items():
                print(f"{metric_name}: {metric.compute()}")

            if self.metrics["accuracy"].value > best_val_accuracy:
                best_val_accuracy = metrics["accuracy"].value
                utils.save_model(model, f"models/{exp_name}.pt")

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        model.eval()

