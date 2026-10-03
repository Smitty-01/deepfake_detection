import torch
from torchvision import transforms
from src.dataset import VideoDataset

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = VideoDataset(
    root="data/processed",
    split_file="splits/train.txt",
    transform=transform
)

video, label = dataset[0]

print("Video shape:", video.shape)
print("Label:", label)
