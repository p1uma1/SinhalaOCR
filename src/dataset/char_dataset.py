import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, processor_name="google/vit-base-patch16-384"):
        self.processor = ViTImageProcessor.from_pretrained(processor_name)
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist.")
            
        self.dataset = ImageFolder(split_dir)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # The processor returns a dict with 'pixel_values', which is a list of tensors
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        
        return pixel_values, label

def get_char_datasets(data_dir, batch_size=32, num_workers=4):
    train_dataset = CharDataset(data_dir, split="train")
    valid_dataset = CharDataset(data_dir, split="valid")
    
    # Try to load test dataset if it exists, otherwise return None
    try:
        test_dataset = CharDataset(data_dir, split="test")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    except FileNotFoundError:
        test_loader = None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    classes = train_dataset.classes
    return train_loader, valid_loader, test_loader, classes
