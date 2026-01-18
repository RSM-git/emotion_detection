
class Metric:
    def compute(self):
        raise NotImplementedError("")

    def reset(self):
        raise NotImplementedError("")

    def update(self, prediction, ground_truth):
        raise NotImplementedError()


class Acurracy(Metric):
    def __init__(self):
        self.correct: int = 0
        self.total: int = 0

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, prediction, ground_truth):
        self.correct += (prediction.argmax(dim=1) == ground_truth).sum()
        self.total += len(ground_truth)

class Precision(Metric):
    def __init__(self, num_classes: int):
        self.tps: list[int] = [0] * num_classes
        self.fps: list[int] = [0] * num_classes

    def compute(self):
        return [self.tps[i] / (self.tps[i] + self.fps[i]) if (self.tps[i] + self.fps[i]) > 0 else 0.0 for i in range(len(self.tps))]

    def reset(self):
        self.tps = [0] * len(self.tps)
        self.fps = [0] * len(self.fps)

    def update(self, prediction, ground_truth):
        for i in range(len(self.tps)):
            self.tps[i] += ((prediction.argmax(dim=1) == i) & (ground_truth == i)).sum().item()
            self.fps[i] += ((prediction.argmax(dim=1) != i) & (prediction.argmax(dim=1) == i)).sum().item()

class Recall(Metric):
    def __init__(self, num_classes: int):
        self.tps: list[int] = [0] * num_classes
        self.fns: list[int] = [0] * num_classes
    
    def compute(self):
        return [self.tps[i] / (self.tps[i] + self.fns[i]) if (self.tps[i] + self.fns[i]) > 0 else 0.0 for i in range(len(self.tps))]
    
    def reset(self):
        self.tps = [0] * len(self.tps)
        self.fns = [0] * len(self.fns)

    def update(self, prediction, ground_truth):
        # The task is multiclass classification
        for i in range(len(self.tps)):
            self.tps[i] += ((prediction.argmax(dim=1) == i) & (ground_truth == i)).sum().item()
            self.fns[i] += ((prediction.argmax(dim=1) != i) & (ground_truth == i)).sum().item()
        