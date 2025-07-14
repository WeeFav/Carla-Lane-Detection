from torch.utils.data import Dataset
import os
from PIL import Image

class DemoDataset(Dataset):
    def __init__(self, img_folder, img_transform=None):
        self.img_transform = img_transform
        self.img_paths = []
        for path in os.listdir(img_folder):
            self.img_paths.append(os.path.join(img_folder, path))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, img_path

    def __len__(self):
        return len(self.img_paths)
    
class LaneDataset(Dataset):
    def __init__(self):
        pass
        
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass