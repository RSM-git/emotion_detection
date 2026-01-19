import torch

class Metric:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def compute(self):
        raise NotImplementedError("")

    def reset(self):
        raise NotImplementedError("")

    def update(self, prediction, ground_truth):
        raise NotImplementedError()


class Accuracy(Metric):
    def __init__(self):
        self.correct: int = 0
        self.total: int = 0
        self.value: float = 0

    def compute(self):
        self.value = self.correct / self.total
        return self.value

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, prediction, ground_truth):
        self.correct += (prediction.argmax(dim=1) == ground_truth).sum()
        self.total += len(ground_truth)

class Precision(Metric):
    def __init__(self, num_classes: int, device: torch.device = torch.device("cpu")):
        self.device = device
        self.tps: torch.Tensor = torch.zeros(num_classes, dtype=torch.int32, device=self.device)
        self.fps: torch.Tensor = torch.zeros(num_classes, dtype=torch.int32, device=self.device)

    def compute(self):
        return self.tps / (self.tps + self.fps).clamp(min=1)

    def reset(self):
        self.tps = torch.zeros_like(self.tps)
        self.fps = torch.zeros_like(self.fps)

    def update(self, prediction, ground_truth):
        preds = prediction.argmax(dim=1)
        arange = torch.arange(len(self.tps), device=preds.device)

        self.tps += ((preds.unsqueeze(1) == arange).T & (ground_truth.unsqueeze(1) == arange).T).sum(dim=1)
        self.fps += ((preds.unsqueeze(1) == arange).T & (ground_truth.unsqueeze(1) != arange).T).sum(dim=1)

class Recall(Metric):
    def __init__(self, num_classes: int, device: torch.device = torch.device("cpu")):
        self.device = device
        self.tps: torch.Tensor = torch.zeros(num_classes, dtype=torch.int32, device=self.device)
        self.fns: torch.Tensor = torch.zeros(num_classes, dtype=torch.int32, device=self.device)

    def compute(self):
        return self.tps / (self.tps + self.fns).clamp(min=1)

    def reset(self):
        self.tps = torch.zeros_like(self.tps)
        self.fns = torch.zeros_like(self.fns)

    def update(self, prediction, ground_truth):
        preds = prediction.argmax(dim=1)
        arange = torch.arange(len(self.tps), device=self.device)

        self.tps += ((preds.unsqueeze(1) == arange).T & (ground_truth.unsqueeze(1) == arange).T).sum(dim=1)
        self.fns += ((preds.unsqueeze(1) != arange).T & (ground_truth.unsqueeze(1) == arange).T).sum(dim=1)
