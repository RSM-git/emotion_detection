import torch

from emotion_detection.metrics import Accuracy, Recall, Precision

def test_accuracy():
    accuracy = Accuracy()
    prediction = torch.tensor([
        [0.1, 0.7, 0.2, 0.0],
        [0.9, 0.05, 0.025, 0.025],
        [0.2, 0.2, 0.5, 0.1],
        [0.3, 0.4, 0.2, 0.1],
    ])

    ground_truth = torch.tensor([1, 0, 2, 1])

    accuracy.update(prediction, ground_truth)
    assert abs(accuracy.compute() - 1.0) < 1e-6

def test_recall():
    recall = Recall(num_classes=4)
    prediction = torch.tensor([
        [0.1, 0.7, 0.2, 0.0],
        [0.9, 0.05, 0.025, 0.025],
        [0.2, 0.2, 0.5, 0.1],
        [0.3, 0.4, 0.2, 0.1],
    ])

    ground_truth = torch.tensor([1, 0, 2, 1])

    recall.update(prediction, ground_truth)
    
    recalls = recall.compute()
    expected_recalls = [1.0, 1.0, 1.0, 0.0]
    for r, er in zip(recalls, expected_recalls):
        assert abs(r - er) < 1e-6

def test_precision():
    precision = Precision(num_classes=4)
    prediction = torch.tensor([
        [0.1, 0.7, 0.2, 0.0],
        [0.9, 0.05, 0.025, 0.025],
        [0.2, 0.2, 0.5, 0.1],
        [0.3, 0.4, 0.2, 0.1],
    ])

    ground_truth = torch.tensor([1, 0, 2, 1])

    precision.update(prediction, ground_truth)
    
    precisions = precision.compute()
    expected_precisions = [1.0, 1.0, 1.0, 0.0]
    for p, ep in zip(precisions, expected_precisions):
        assert abs(p - ep) < 1e-6