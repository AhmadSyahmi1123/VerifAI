# v4_convnext_tiny.py (with pause/resume support)

import os
import math
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ----------------------------
# Dataset wrapper
# ----------------------------
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.dataset = ImageFolder(folder)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = np.array(self.dataset.loader(path))
        img = self.transform(image=img)["image"]
        return img, label


# ----------------------------
# EMA helper
# ----------------------------
class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = {}
        self.collected_names = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
                self.collected_names.append(name)
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                new = p.detach()
                self.shadow[name] = self.shadow[name].to(new.device)  # ensure same device
                self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)


    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().cpu().clone()
                p.data.copy_(self.shadow[name].to(p.device))

    def restore(self, model):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}

    def to(self, device):
        for name, tensor in self.shadow.items():
            self.shadow[name] = tensor.to(device)


# ----------------------------
# Config
# ----------------------------
DATASET_DIR = "dataset_split"
IMG_SIZE = 256
MODEL_NAME = "convnext_small"
BATCH_SIZE = 32
ACCUM_STEPS = 2
LR = 1e-4
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔄 NEW: Resume checkpoint path
RESUME_PATH = os.path.join(SAVE_DIR, "resume_training.pth")


# ----------------------------
# Transforms
# ----------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    ToTensorV2(),
], p=1.0)

val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
], p=1.0)


# ----------------------------
# Training
# ----------------------------
def train():
    print("Device:", DEVICE, "Num workers:", NUM_WORKERS)
    torch.backends.cudnn.benchmark = True

    train_ds = AlbumentationsDataset(os.path.join(DATASET_DIR, "train"), train_transform)
    val_ds = AlbumentationsDataset(os.path.join(DATASET_DIR, "val"), val_transform)

    labels = [y for _, y in train_ds.dataset.samples]
    class_sample_count = np.array([labels.count(c) for c in sorted(set(labels))])
    weights = np.array([1.0 / class_sample_count[label] for label in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # Model
    model = create_model(MODEL_NAME, pretrained=True, num_classes=1, drop_rate=0.1, drop_path_rate=0.05)
    model.to(DEVICE)
    model.to(memory_format=torch.channels_last)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    total_steps = math.ceil(len(train_loader) * EPOCHS / ACCUM_STEPS)
    def lr_lambda(step):
        if step < max(1, int(0.03 * total_steps)):
            return float(step) / float(max(1, int(0.03 * total_steps)))
        progress = (step - int(0.03 * total_steps)) / max(1, total_steps - int(0.03 * total_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, decay=0.9998)

    # 🔄 NEW: Resume variables
    start_epoch = 1
    global_step = 0
    best_val_acc = 0.0

    # ----------------------------
    # 🔄 LOAD RESUME CHECKPOINT IF EXISTS
    # ----------------------------
    if os.path.exists(RESUME_PATH):
        print("🔄 Resuming training from:", RESUME_PATH)
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        ema.shadow = ckpt["ema_shadow"]
        ema.to(DEVICE)

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_acc = ckpt["best_val_acc"]

        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)
        optimizer.zero_grad()

        for step, (images, labels) in pbar:
            images = images.to(DEVICE).to(memory_format=torch.channels_last)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
                ema.update(model)

            running_loss += (loss.item() * ACCUM_STEPS)
            pbar.set_postfix({"train_loss": f"{running_loss/(step+1):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        ema.apply_shadow(model)
        val_correct = 0
        val_total = 0
        val_loss = 0.0
 
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", ncols=120):
                images = images.to(DEVICE).to(memory_format=torch.channels_last)
                labels = labels.float().unsqueeze(1).to(DEVICE)
 
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss_v = criterion(outputs, labels)
                    probs = torch.sigmoid(outputs)

                preds = (probs > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)
                val_loss += loss_v.item() * labels.size(0)

        ema.restore(model)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch} -> Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ema.apply_shadow(model)

            # Save FP16 weights
            state = {k: v.cpu().half() for k, v in model.state_dict().items()}
            torch.save(state, os.path.join(SAVE_DIR, "convnext_tiny_best_fp16.pth"))

            ema.restore(model)
            print("✅ Saved best FP16 checkpoint!")

        # ----------------------------
        # 🔄 SAVE RESUME CHECKPOINT
        # ----------------------------
        resume_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema_shadow": ema.shadow,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_acc": best_val_acc,
        }
        torch.save(resume_ckpt, RESUME_PATH)
        print("💾 Saved resumable checkpoint.")

    print("Training complete. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    train()
