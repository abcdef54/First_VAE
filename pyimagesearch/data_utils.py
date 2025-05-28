from torch.utils.data import Dataset

import glob
from PIL import Image


class CelebADataset(Dataset):
    def __init__(self, root: str, transform = None) -> None:
        self.root = root
        self.transform = transform
        self.all_images = list(glob.iglob(root + '/*.jpg'))

    def __len__(self) -> int:
        return len(self.all_images)
    
    def __getitem__(self, idx: int):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert(('RGB'))
        if self.transform:
            image = self.transform(image)
        return image
