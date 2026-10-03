import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root, split_file, transform=None):
        self.root = root
        self.transform = transform

        with open(split_file) as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = 1 if path.startswith("fake") else 0

        frames_dir = os.path.join(self.root, path)

        frames = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # SKIP BAD VIDEOS
        if len(frames) < 30:
            # try next sample (wrap around)
            return self.__getitem__((idx + 1) % len(self.samples))

        frames = frames[:30]

        images = []
        for f in frames:
            img = Image.open(os.path.join(frames_dir, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        video = torch.stack(images)
        return video, torch.tensor(label, dtype=torch.float32)
