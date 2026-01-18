import torch
import torch.nn as nn

from emotion_detection.model import EmotionNet
from emotion_detection.dataset import get_dataloader
from emotion_detection.metrics import Accuracy, Recall, Precision

from tqdm import tqdm

train_dataloader = get_dataloader()
test_dataloader = get_dataloader(train=False)


model = EmotionNet()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

epochs: int = 100

metrics = [Accuracy(), Precision(num_classes=7, device=torch.device("cuda")), Recall(num_classes=7, device=torch.device("cuda"))]

for epoch in tqdm(range(epochs)):
    
    model.train()

    for metric in metrics:
        metric.reset()

    for batch in train_dataloader:
        img = batch["data"].cuda()
        label = batch["label"].cuda()
        
        out = model(img)

        loss = criterion(out, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for metric in metrics:
            metric.update(out, label)
        
    print(f"Epoch {epoch+1}/{epochs} Train Metrics:")
    for metric in metrics:
        print(f"{metric}: {metric.compute()}")

    model.eval()
    for metric in metrics:
        metric.reset()
    
    with torch.no_grad():
        for batch in test_dataloader:
            img = batch["data"].cuda()
            label = batch["label"].cuda()
            
            out = model(img)
            print(out.argmax(dim=1))
            print(nn.functional.softmax(out, dim=1))

            for metric in metrics:
                metric.update(out, label)
        
    print(f"Epoch {epoch+1}/{epochs} Test Metrics:")

    for metric in metrics:
        print(f"{metric}: {metric.compute()}")

    
