import os
import torch
import torch.nn as nn
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================
# DATASET
# ======================================

class ChangeDataset(Dataset):
    def __init__(self, base_path):
        self.samples = []

        img_root = os.path.join(
            base_path,
            "Onera Satellite Change Detection dataset - Images"
        )

        for city in os.listdir(img_root):
            pair_path = os.path.join(img_root, city, "pair")

            img1 = os.path.join(pair_path, "img1.png")
            img2 = os.path.join(pair_path, "img2.png")

            if os.path.exists(img1) and os.path.exists(img2):
                self.samples.append((img1, img2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path = self.samples[idx]

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        img1 = cv2.resize(img1, (256,256))
        img2 = cv2.resize(img2, (256,256))

        img1 = img1.transpose(2,0,1) / 255.0
        img2 = img2.transpose(2,0,1) / 255.0

        x = np.concatenate([img1, img2], axis=0)

        # pseudo label → difference magnitude
        y = np.mean(np.abs(img1 - img2), axis=0, keepdims=True)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ======================================
# MODEL
# ======================================

class ChangeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4)
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


# ======================================
# TRAINING
# ======================================

def train_deep_model():

    base = "data/raw/oscd"

    dataset = ChangeDataset(base)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ChangeCNN().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    print("Training Deep Change Detector...")

    for epoch in range(5):

        total_loss = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss:", total_loss)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/deep_change_cnn.pth")

    print("Deep model saved.")


if __name__ == "__main__":
    train_deep_model()