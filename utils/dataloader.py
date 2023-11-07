import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os

# Builds a dataset with the specified image file-paths.
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.image_paths = []
        print(self.img_paths)
        self.filenames=os.listdir(self.img_paths)
        
        for path_format in self.filenames:
            self.image_paths .append( os.path.join(img_paths,path_format))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # dummy label
        return image, -1
    

# Prefixes the index and drops the label for each sample.
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index][0]
    
def get_dataset(args):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    print("your dataset is :",args.img_paths)
    dataset = ImageDataset(args.img_paths, transform=transform)

    num_generation = len(dataset) if args.num_generation == -1 else args.num_generation
    if args.chunk is not None:
        num_chunks, chunk_index = args.chunk
        chunk_size = int(np.ceil(num_generation / num_chunks))
        chunk_start = chunk_size * chunk_index
        chunk_end = min(chunk_start + chunk_size, num_generation)
    else:
        chunk_start = 0
        chunk_end = num_generation

    dataset = IndexedDataset(dataset)
    dataset = Subset(dataset, range(chunk_start, chunk_end))

    return dataset