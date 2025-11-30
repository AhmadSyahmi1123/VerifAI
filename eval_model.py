import os
import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# Dataset with Albumentations
# ---------------------------
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.dataset = ImageFolder(folder)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = self.dataset.loader(path)
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img, label


# ---------------------------
#         MAIN EVAL
# ---------------------------

def evaluate():
    DATASET_DIR = "dataset_split"
    MODEL_PATH = "models\convnext_tiny_v13.pth"
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", DEVICE)

    # --- transforms ---
    test_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])

    # --- dataset ---
    test_ds = AlbumentationsDataset(os.path.join(DATASET_DIR, "test"), test_transform)
    test_loader = DataLoader(test_ds, batch_size=150, shuffle=False, num_workers=2)

    # --- recreate the model arch ---
    model = create_model("convnext_small", pretrained=True, num_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_labels = []

    total_loss = 0
    total_samples = 0
    correct = 0

    # ---------------------------
    #        Evaluation loop
    # ---------------------------
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # accumulate loss
            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            # prediction
            preds = (torch.sigmoid(outputs) > 0.5).long()

            correct += (preds == labels.long()).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100

    # ---------------------------
    #       Print results
    # ---------------------------
    print("\n✅ Evaluation Results")
    print("-------------------------")
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    target_names = ["real (0)", "fake (1)"]
    print(classification_report(all_labels, all_preds, target_names=target_names))


if __name__ == "__main__":
    evaluate()
