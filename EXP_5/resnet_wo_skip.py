import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 1. DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 2. DIRECTORIES FOR OUTPUTS
# ============================================================
OUTPUT_DIR = "outputs_resnet_wo_skip"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

# ============================================================
# 3. HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
NUM_WORKERS = 0

# ============================================================
# 4. DATA LOADING (SVHN Format 2 via torchvision)
# ============================================================
DATA_DIR = os.path.join("dataset", "SVHN")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                         std=[0.1980, 0.2010, 0.1970])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                         std=[0.1980, 0.2010, 0.1970])
])

print("Downloading/Loading SVHN dataset (Format 2 - cropped 32x32)...")
train_ds = datasets.SVHN(root=DATA_DIR, split="train", download=True, transform=transform_train)
test_ds = datasets.SVHN(root=DATA_DIR, split="test", download=True, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

# ============================================================
# 5. VISUALIZE SAMPLE IMAGES
# ============================================================
def visualize_samples(dataset, title, save_path):
    """Show a grid of sample images from the dataset."""
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1)
    std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1)

    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        img = img * std + mean
        img = img.clamp(0, 1)
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"Label: {label}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

visualize_samples(train_ds, "SVHN Training Samples",
                  os.path.join(OUTPUT_DIR, "plots", "sample_images.png"))

# ============================================================
# 6. RESNET WITHOUT SKIP CONNECTIONS
# ============================================================
class PlainBlock(nn.Module):
    """Plain residual block WITHOUT skip connections."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # NO skip connection — identity shortcut is removed entirely

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))  # no addition of identity
        return out


class ResNetWithoutSkip(nn.Module):
    """
    Plain ResNet (no skip connections) for SVHN (32x32 input).
    Same depth and channel sizes as ResNetWithSkip for fair comparison.

    Structure:
        - Initial conv layer
        - Stage 1: 3 PlainBlocks, 64 channels
        - Stage 2: 3 PlainBlocks, 128 channels (stride=2 -> 16x16)
        - Stage 3: 3 PlainBlocks, 256 channels (stride=2 -> 8x8)
        - Global Average Pool -> FC -> 10 classes
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 64 channels, 32x32
        self.stage1_block1 = PlainBlock(64, 64)
        self.stage1_block2 = PlainBlock(64, 64)
        self.stage1_block3 = PlainBlock(64, 64)

        # Stage 2: 128 channels, 16x16
        self.stage2_block1 = PlainBlock(64, 128, stride=2)
        self.stage2_block2 = PlainBlock(128, 128)
        self.stage2_block3 = PlainBlock(128, 128)

        # Stage 3: 256 channels, 8x8
        self.stage3_block1 = PlainBlock(128, 256, stride=2)
        self.stage3_block2 = PlainBlock(256, 256)
        self.stage3_block3 = PlainBlock(256, 256)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Weight initialization (Kaiming/He initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)

        # Stage 1
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.stage1_block3(x)

        # Stage 2
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_block3(x)

        # Stage 3
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)

        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# 7. CREATE MODEL & PRINT ARCHITECTURE
# ============================================================
model = ResNetWithoutSkip(num_classes=NUM_CLASSES).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResNet WITHOUT Skip Connections — {total_params:,} parameters")

# ============================================================
# 8. EXPORT MODEL TO ONNX FOR NETRON VISUALIZATION
# ============================================================
def export_to_onnx(model, save_path, input_size=(1, 3, 32, 32)):
    """Export model to ONNX format for Netron visualization."""
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX model exported to: {save_path}")
    print(f"Open with Netron: run 'netron \"{save_path}\"' in terminal")

onnx_path = os.path.join(OUTPUT_DIR, "models", "resnet_wo_skip.onnx")
export_to_onnx(model, onnx_path)
print()

# ============================================================
# 9. GRADIENT & WEIGHT TRACKING SETUP
# ============================================================
TRACKED_LAYERS = {
    "initial.0":           model.initial[0],
    "stage1_block1.conv1": model.stage1_block1.conv1,
    "stage1_block3.conv2": model.stage1_block3.conv2,
    "stage2_block1.conv1": model.stage2_block1.conv1,
    "stage2_block3.conv2": model.stage2_block3.conv2,
    "stage3_block1.conv1": model.stage3_block1.conv1,
    "stage3_block3.conv2": model.stage3_block3.conv2,
}

history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
    "epoch_grad_norms": {name: [] for name in TRACKED_LAYERS},
    "batch_grad_norms": {name: [] for name in TRACKED_LAYERS},
    "weight_norms": {name: [] for name in TRACKED_LAYERS},
    "weight_deltas": {name: [] for name in TRACKED_LAYERS},
}

prev_weights = {}
for name, layer in TRACKED_LAYERS.items():
    prev_weights[name] = layer.weight.data.clone()


def log_gradients(batch_grad_accumulator):
    for name, layer in TRACKED_LAYERS.items():
        if layer.weight.grad is not None:
            grad_norm = layer.weight.grad.data.norm(2).item()
            batch_grad_accumulator[name].append(grad_norm)


def log_weight_norms():
    for name, layer in TRACKED_LAYERS.items():
        w_norm = layer.weight.data.norm(2).item()
        history["weight_norms"][name].append(w_norm)


def log_weight_deltas():
    for name, layer in TRACKED_LAYERS.items():
        delta = (layer.weight.data - prev_weights[name]).norm(2).item()
        history["weight_deltas"][name].append(delta)
        prev_weights[name] = layer.weight.data.clone()


# ============================================================
# 10. TRAINING & EVALUATION FUNCTIONS
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_grad_accumulator = defaultdict(list)

    pbar = tqdm(train_loader, desc=f"  Train", leave=False, unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        log_gradients(batch_grad_accumulator)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    for name in TRACKED_LAYERS:
        batch_norms = batch_grad_accumulator[name]
        epoch_mean = np.mean(batch_norms) if batch_norms else 0.0
        history["epoch_grad_norms"][name].append(epoch_mean)
        history["batch_grad_norms"][name].extend(batch_norms)

    log_weight_norms()
    log_weight_deltas()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(epoch_acc)

    return epoch_loss, epoch_acc


def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"  Eval ", leave=False, unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(acc=f"{100.*correct/total:.1f}%")

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total
    history["test_loss"].append(epoch_loss)
    history["test_acc"].append(epoch_acc)

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ============================================================
# 11. TRAINING LOOP
# ============================================================
print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)

best_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]  LR: {scheduler.get_last_lr()[0]:.6f}")
    print("-" * 40)

    train_loss, train_acc = train_one_epoch(epoch)
    test_loss, test_acc, all_preds, all_labels = evaluate()
    scheduler.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

    print(f"  Gradient Norms (L2):")
    for name in TRACKED_LAYERS:
        grad_val = history["epoch_grad_norms"][name][-1]
        weight_val = history["weight_norms"][name][-1]
        delta_val = history["weight_deltas"][name][-1]
        print(f"    {name:30s} | Grad: {grad_val:.6f} | "
              f"Weight: {weight_val:.4f} | Delta: {delta_val:.6f}")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR, "models", "resnet_wo_skip_best.pth"))
        print(f"  >> New best model saved! Acc: {best_acc:.2f}%")

print("\n" + "=" * 60)
print(f"TRAINING COMPLETE | Best Test Accuracy: {best_acc:.2f}%")
print("=" * 60)

torch.save(model.state_dict(),
           os.path.join(OUTPUT_DIR, "models", "resnet_wo_skip_final.pth"))

# ============================================================
# 12. SAVE TRAINING HISTORY
# ============================================================
history_json = {
    "train_loss": history["train_loss"],
    "train_acc": history["train_acc"],
    "test_loss": history["test_loss"],
    "test_acc": history["test_acc"],
    "epoch_grad_norms": {k: v for k, v in history["epoch_grad_norms"].items()},
    "weight_norms": {k: v for k, v in history["weight_norms"].items()},
    "weight_deltas": {k: v for k, v in history["weight_deltas"].items()},
}
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
    json.dump(history_json, f, indent=2)
print(f"Training history saved to {OUTPUT_DIR}/training_history.json")


# ============================================================
# 13. PLOTTING FUNCTIONS
# ============================================================
def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs_range, history["test_loss"], "r-o", label="Test Loss", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("ResNet WITHOUT Skip Connections — Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy_curves():
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax.plot(epochs_range, history["test_acc"], "r-o", label="Test Acc", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet WITHOUT Skip Connections — Accuracy Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "accuracy_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_epoch_gradient_norms():
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs_range = range(1, EPOCHS + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(TRACKED_LAYERS)))
    for (name, _), color in zip(TRACKED_LAYERS.items(), colors):
        ax.plot(epochs_range, history["epoch_grad_norms"][name],
                "-o", label=name, color=color, markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gradient Norm (L2)", fontsize=12)
    ax.set_title("ResNet WITHOUT Skip — Per-Epoch Gradient Norms (L2)\n"
                 "(Without skip connections, early layer gradients may vanish)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "epoch_gradient_norms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_batch_gradient_norms():
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(TRACKED_LAYERS)))
    for (name, _), color in zip(TRACKED_LAYERS.items(), colors):
        norms = history["batch_grad_norms"][name]
        window = 50
        if len(norms) > window:
            smoothed = np.convolve(norms, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, color=color, alpha=0.8, linewidth=1)
        else:
            ax.plot(norms, label=name, color=color, alpha=0.8, linewidth=1)
    ax.set_xlabel("Batch (smoothed, window=50)", fontsize=12)
    ax.set_ylabel("Gradient Norm (L2)", fontsize=12)
    ax.set_title("ResNet WITHOUT Skip — Per-Batch Gradient Norms (Smoothed)\n"
                 "(Decreasing gradient norms in early layers = vanishing gradient)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "batch_gradient_norms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_weight_norms():
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs_range = range(1, EPOCHS + 1)
    colors = plt.cm.plasma(np.linspace(0, 1, len(TRACKED_LAYERS)))
    for (name, _), color in zip(TRACKED_LAYERS.items(), colors):
        ax.plot(epochs_range, history["weight_norms"][name],
                "-s", label=name, color=color, markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weight Norm (L2)", fontsize=12)
    ax.set_title("ResNet WITHOUT Skip — Weight Norms Over Training",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "weight_norms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_weight_deltas():
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs_range = range(1, EPOCHS + 1)
    colors = plt.cm.inferno(np.linspace(0.2, 0.9, len(TRACKED_LAYERS)))
    for (name, _), color in zip(TRACKED_LAYERS.items(), colors):
        ax.plot(epochs_range, history["weight_deltas"][name],
                "-^", label=name, color=color, markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weight Change (L2 of Δw)", fontsize=12)
    ax.set_title("ResNet WITHOUT Skip — Weight Updates Per Epoch\n"
                 "(Early layers may receive tiny updates → vanishing gradient effect)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "weight_deltas.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_gradient_heatmap():
    layer_names = list(TRACKED_LAYERS.keys())
    data = np.array([history["epoch_grad_norms"][name] for name in layer_names])
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_xticks(range(EPOCHS))
    ax.set_xticklabels(range(1, EPOCHS + 1), fontsize=9)
    ax.set_title("ResNet WITHOUT Skip — Gradient Norm Heatmap (Layers × Epochs)\n"
                 "(Darker early rows = vanishing gradients in early layers)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Gradient Norm (L2)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "gradient_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=[str(i) for i in range(10)])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("ResNet WITHOUT Skip — Confusion Matrix (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_sample_predictions(dataset, preds, labels, num_samples=16):
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle("ResNet WITHOUT Skip — Sample Predictions (Test Set)",
                 fontsize=14, fontweight="bold")

    mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(3, 1, 1)
    std = torch.tensor([0.1980, 0.2010, 0.1970]).view(3, 1, 1)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        img, true_label = dataset[idx]
        img_display = img * std + mean
        img_display = img_display.clamp(0, 1)

        pred_label = preds[idx]
        color = "green" if pred_label == true_label else "red"

        ax.imshow(img_display.permute(1, 2, 0).numpy())
        ax.set_title(f"T:{true_label} P:{pred_label}", fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "sample_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_gradient_ratio():
    layer_names = list(TRACKED_LAYERS.keys())
    first_layer = layer_names[0]
    last_layer = layer_names[-1]

    first_grads = np.array(history["epoch_grad_norms"][first_layer])
    last_grads = np.array(history["epoch_grad_norms"][last_layer])

    ratio = first_grads / (last_grads + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, ratio, "g-o", markersize=5, linewidth=2)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Ideal ratio = 1.0")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"Gradient Ratio ({first_layer} / {last_layer})", fontsize=11)
    ax.set_title("ResNet WITHOUT Skip — Gradient Ratio (First Layer / Last Layer)\n"
                 "(Ratio << 1.0 = vanishing gradients in early layers)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plots", "gradient_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 14. GENERATE ALL PLOTS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

plot_loss_curves()
plot_accuracy_curves()
plot_epoch_gradient_norms()
plot_batch_gradient_norms()
plot_weight_norms()
plot_weight_deltas()
plot_gradient_heatmap()
plot_confusion_matrix(all_preds, all_labels)
plot_sample_predictions(test_ds, all_preds, all_labels)
plot_gradient_ratio()

# ============================================================
# 15. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY: ResNet WITHOUT Skip Connections")
print("=" * 60)
print(f"Model: Plain ResNet-20 style (no skip connections)")
print(f"Dataset: SVHN (73,257 train / 26,032 test)")
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Test Loss: {history['test_loss'][-1]:.4f}")
print()
print("Key Observations (No Skip Connections):")
print("  ✗ Gradients weaken significantly in early layers")
print("  ✗ Early layer weights receive smaller updates")
print("  ✗ Gradient ratio (first/last layer) may drop well below 1.0")
print()
print(f"All outputs saved to: {OUTPUT_DIR}/")
print(f"  plots/               — All visualizations")
print(f"  models/              — ONNX + PyTorch weights")
print(f"  training_history.json — Full training logs")
print()
print("To visualize model architecture in Netron, run:")
print(f'  netron "{os.path.abspath(onnx_path)}"')
print("=" * 60)