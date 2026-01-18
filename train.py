import torch
import torch.nn as nn

from src.model import EmotionNet
from src.dataset import get_dataloader

from tqdm import tqdm

train_dataloader = get_dataloader()
test_dataloader = get_dataloader(train=False)


model = EmotionNet()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

epochs: int = 100



for epoch in tqdm(range(epochs)):
    correct = 0
    total = 0
    for batch in train_dataloader:
        img = batch["data"].cuda()
        label = batch["label"].cuda()
        
        out = model(img)

        loss = criterion(out, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += (out.argmax(dim=1) == label).sum()
        total += len(label)

    print(correct/total)
