from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

class DogDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        self.tensor_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((127, 127)),
        ])

        self.processing_transform = transforms.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=(127, 127), scale=(0.75, 1), antialias=True),
        ])
    
    def __len__(self):
        return len(os.listdir(self.dataset_dir))
    
    def __getitem__(self, index):
        image_path = self.dataset_dir + "/" + os.listdir(self.dataset_dir)[index]
        image = Image.open(image_path)
        image = self.tensor_transform(image).float()
        image = self.processing_transform(image)
        return image