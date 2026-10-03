from loaders import get_loaders

if __name__ == '__main__':
    train_dl, val_dl = get_loaders(batch_size=2, num_workers=0)
    videos, labels = next(iter(train_dl))
    print("Batch video shape:", videos.shape)
    print("Batch labels:", labels)
