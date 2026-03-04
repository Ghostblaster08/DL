"""
train_compare.py
Trains the CNN model with 3 different loss functions and saves each model:
  1. CrossEntropyLoss  (standard multi-class)
  2. BCELoss           (binary, needs sigmoid + one-hot labels)
  3. BCEWithLogitsLoss (binary with built-in sigmoid + one-hot labels)
"""

import os
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized since it had shape")

import torch
from torch import nn
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import UrbanSoundDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

# ── Hyper-parameters ─────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 10
LEARNING_RATE = 0.001
NUM_CLASSES  = 10

# ── Device ───────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}\n")


# ── Model ─────────────────────────────────────────────────────
# NOTE: No Softmax in forward() — CrossEntropyLoss does it internally.
#       For BCE variants, we apply sigmoid in the training loop.
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(128 * 5 * 4, NUM_CLASSES)
        # No softmax here — handled per loss function

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        return self.linear(x)   # raw logits


# ── Data ──────────────────────────────────────────────────────
def get_dataloader():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel,
                            SAMPLE_RATE, NUM_SAMPLES, device)
    return DataLoader(usd, batch_size=BATCH_SIZE, shuffle=True)


# ── Training functions ────────────────────────────────────────
def train_ce(model, loader, optimizer, loss_fn):
    """CrossEntropyLoss — labels are class indices (Long)"""
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_bce(model, loader, optimizer, loss_fn, with_logits=False):
    """BCELoss / BCEWithLogitsLoss — labels need one-hot encoding (Float)"""
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels_onehot = F.one_hot(labels.long(), NUM_CLASSES).float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if not with_logits:
            outputs = torch.sigmoid(outputs)  # BCELoss needs probabilities
        loss = loss_fn(outputs, labels_onehot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Main loop ─────────────────────────────────────────────────
def run_experiment(loss_name):
    print(f"\n{'='*55}")
    print(f"  Training with: {loss_name}")
    print(f"{'='*55}")

    loader = get_dataloader()
    model  = CNNNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if loss_name == "CrossEntropyLoss":
        loss_fn   = nn.CrossEntropyLoss()
        train_fn  = lambda m, l, o, lf: train_ce(m, l, o, lf)
    elif loss_name == "BCELoss":
        loss_fn   = nn.BCELoss()
        train_fn  = lambda m, l, o, lf: train_bce(m, l, o, lf, with_logits=False)
    else:  # BCEWithLogitsLoss
        loss_fn   = nn.BCEWithLogitsLoss()
        train_fn  = lambda m, l, o, lf: train_bce(m, l, o, lf, with_logits=True)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_fn(model, loader, optimizer, loss_fn)
        print(f"  Epoch {epoch:>2}/{EPOCHS}  |  Loss: {avg_loss:.4f}")

    save_path = f"saved_model/soundclassifier_{loss_name}.pth"
    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n  Model saved → {save_path}")
    return save_path


if __name__ == "__main__":
    loss_functions = ["CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"]
    for lf in loss_functions:
        run_experiment(lf)
    print("\n\nAll experiments done! Run `python test_compare.py` to see results.")
