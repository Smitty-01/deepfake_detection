import os
import random

ROOT = "data/processed"
OUT = "splits"
os.makedirs(OUT, exist_ok=True)

random.seed(42)

def split(videos, train=0.7, val=0.15):
    n = len(videos)
    return (
        videos[:int(train*n)],
        videos[int(train*n):int((train+val)*n)],
        videos[int((train+val)*n):]
    )

real = sorted(os.listdir(f"{ROOT}/real"))
fake = sorted(os.listdir(f"{ROOT}/fake"))

random.shuffle(real)
random.shuffle(fake)

real_train, real_val, real_test = split(real)
fake_train, fake_val, fake_test = split(fake)

def save(name, r, f):
    with open(f"{OUT}/{name}.txt", "w") as fp:
        for v in r:
            fp.write(f"real/{v}\n")
        for v in f:
            fp.write(f"fake/{v}\n")

save("train", real_train, fake_train)
save("val", real_val, fake_val)
save("test", real_test, fake_test)

print("Splits created successfully")
