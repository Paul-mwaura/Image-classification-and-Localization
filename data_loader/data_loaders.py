from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image


class TBCDataset(Dataset):
    def __init__(self, data_dir, trsfm):
        self.imgs = [path for path in Path(data_dir).rglob('*.png')]
        self.train = pd.read_csv('train.csv')
        self.trsfm = trsfm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        label = int(self.train[self.train.ID ==
                               self.imgs[idx].name.split('.')[0]]['LABEL'])
        img = self.trsfm(img)
        return img, label


class TBCDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.3, num_workers=1, training=None):
        trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = TBCDataset(data_dir=data_dir, trsfm=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
