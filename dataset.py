import os

from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)

        self.zebra_length = len(self.zebra_images)
        self.horse_length = len(self.horse_images)
        self.dataset_length = max(self.zebra_length, self.horse_length)

    def __len__(self):
        self.dataset_length

    def __getitem__(self, idx: int):
        zebra_image_path = self.zebra_images[idx % self.zebra_length]
        horse_image_path = self.horse_images[idx % self.horse_length]

        zebra_path = os.path.join(self.root_zebra, zebra_image_path)
        horse_path = os.path.join(self.root_horse, horse_image_path)

        zebra_img = np.array(Image.open(zebra_path).convert('RGB'))
        horse_img = np.array(Image.open(horse_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations['image']
            horse_img = augmentations['image0']

        return zebra_img, horse_img
