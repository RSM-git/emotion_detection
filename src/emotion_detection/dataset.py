import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image

from emotion_detection.data import download_dataset

def load_data(train: bool = True):
    if train:
        data_path = download_dataset() / "train" 
    else:
        data_path = download_dataset() / "test"

    return [(img, i) for i, directory in enumerate(data_path.glob("*")) for img in directory.glob("*")]

class EmotionDataset(Dataset):
    def __init__(self, train: bool = True):
        self.samples = load_data(train)
        
        self.samples = [(read_image(img_path), label) for img_path, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]

        image = image.type(torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return {"data": image, "label": label}

def get_dataloader(train: bool = True):
    dataset = EmotionDataset(train)
    return DataLoader(dataset, batch_size=32)

if __name__ == '__main__':
    pass 
