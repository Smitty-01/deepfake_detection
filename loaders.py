import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dataset import VideoDataset


def get_loaders(
    data_root=None,
    train_split=None,
    val_split=None,
    batch_size=4,
    num_workers=2
):
    if data_root is None:
        data_root = os.path.join(PROJECT_ROOT, "data", "processed")
    if train_split is None:
        train_split = os.path.join(PROJECT_ROOT, "splits", "train.txt")
    if val_split is None:
        val_split = os.path.join(PROJECT_ROOT, "splits", "val.txt")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = VideoDataset(
        root=data_root,
        split_file=train_split,
        transform=train_tf
    )

    val_ds = VideoDataset(
        root=data_root,
        split_file=val_split,
        transform=val_tf
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dl, val_dl
