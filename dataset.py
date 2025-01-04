import os
import torch
from torch import Tensor, Type
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from datasets import load_dataset

class COCO2017Dataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable):
        assert split in ["train", "test", "val"]
        data_dir = Path(data_path) / "COCO" / f"{split}2017"     
        self.dataset = load_dataset("imagefolder", data_dir=data_dir)['train']
        self.transforms = transform
        
        print(f"Total images in dataset split: {split}: {len(self.imgs)}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        img = image.convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, 0.0 # dummy datat to prevent breaking

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        assert split in ["train", "test", "val"]
        self.data_dir = Path(data_path) / "OxfordPets"   / "images"     
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        if split == "train":
            self.imgs = imgs[:int(len(imgs) * 0.75)]
        elif split == "val":
            self.imgs = imgs[int(len(imgs) * 0.75):int(len(imgs) * 0.85)]
        elif split == "test":
            self.imgs = imgs[int(len(imgs) * 0.85):]
    
        print(f"Total images in dataset split: {split}: {len(self.imgs)}")

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset: str = 'OxfordPets',
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        assert dataset in ['OxfordPets', 'CelebA', 'COCO2017']
        if dataset == 'OxfordPets':
            self.dataset_cls = OxfordPets
        elif dataset == 'CelebA':
            self.dataset_cls = MyCelebA
        elif dataset == 'COCO2017':
            self.dataset_cls = COCO2017Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(self.patch_size),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(self.patch_size),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        test_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(self.patch_size),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        print("setup train dataset")
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            split='train',
            transform=train_transforms,
        )
        
        print("setup val  dataset")
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            split='valid' if self.dataset_cls == MyCelebA else 'val',
            transform=val_transforms,
        )

        print("setup test  dataset")
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            split='test',
            transform=test_transforms,
        )
        print("setup completed")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
     