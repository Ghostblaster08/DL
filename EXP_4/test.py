"""
test_compare.py
Evaluates and compares the 3 saved models (one per loss function).
Prints accuracy, precision, recall, F1, and a per-class report for each.
Also generates visualisation plots:
  1. Grouped bar chart  – Accuracy / Precision / Recall / F1 across models
  2. Per-class F1 comparison bar chart
  3. Confusion matrices (one per model, side-by-side)
  4. Radar / spider chart for overall metric comparison
"""

import os
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized since it had shape")

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, f1_score as sk_f1)

from dataset import UrbanSoundDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

# ── Constants ─────────────────────────────────────────────────
NUM_CLASSES = 10
CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]
CLASS_SHORT = ["AirCond", "CarHorn", "Children", "DogBark",
               "Drilling", "EngIdle", "GunShot", "Jackhmr", "Siren", "StrMusic"]

LOSS_COLORS = {
    "CrossEntropyLoss":  "#4C72B0",
    "BCELoss":           "#DD8452",
    "BCEWithLogitsLoss": "#55A868",
}

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Device ───────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}\n")


# ── Model (no softmax) ───────────────────────────────────────
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(128 * 5 * 4, NUM_CLASSES)

    def forward(self, x):
        return self.linear(self.flatten(self.conv4(self.conv3(self.conv2(self.conv1(x))))))


# ── Dataset ──────────────────────────────────────────────────
def get_dataloader():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel, SAMPLE_RATE, NUM_SAMPLES, device)
    return usd, len(usd)


# ── Evaluate one model ───────────────────────────────────────
def evaluate(model_path, loss_name):
    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_path} not found. Run train_compare.py first.\n")
        return None

    model = CNNNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    from torch.utils.data import DataLoader
    usd, n = get_dataloader()
    loader = DataLoader(usd, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    per_class_f1 = sk_f1(all_labels, all_preds, average=None,
                          labels=list(range(NUM_CLASSES)), zero_division=0)

    return {
        "loss": loss_name, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "preds": all_preds, "labels": all_labels, "per_class_f1": per_class_f1,
    }


# ═══════════════════════════════════════════════════════════════
# ── Plotting helpers ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def _style():
    """Apply a clean dark-grid style globally."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#F7F7F7",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DEDEDE",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })


# ── Plot 1: Grouped bar chart – overall metrics ───────────────
def plot_metric_comparison(results):
    _style()
    metrics = ["acc", "prec", "rec", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    loss_names = [r["loss"] for r in results]
    n_metrics  = len(metrics)
    n_models   = len(results)
    x = np.arange(n_metrics)
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, r in enumerate(results):
        vals  = [r[m] * 100 for m in metrics]
        color = LOSS_COLORS[r["loss"]]
        bars  = ax.bar(x + i * width, vals, width, label=r["loss"],
                       color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#333")

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Overall Metric Comparison Across Loss Functions", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "1_metric_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# ── Plot 2: Per-class F1 grouped bar chart ────────────────────
def plot_per_class_f1(results):
    _style()
    x = np.arange(NUM_CLASSES)
    n_models = len(results)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, r in enumerate(results):
        color = LOSS_COLORS[r["loss"]]
        ax.bar(x + i * width, r["per_class_f1"] * 100, width,
               label=r["loss"], color=color, alpha=0.88,
               edgecolor="white", linewidth=0.7)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(CLASS_SHORT, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "2_per_class_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# ── Plot 3: Confusion matrices side by side ───────────────────
def plot_confusion_matrices(results):
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True,
                          "axes.grid": False})
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["labels"], r["preds"], labels=list(range(NUM_CLASSES)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(r["loss"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_SHORT, fontsize=8)

        thresh = 0.5
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if val > thresh else "#222")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Normalised Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "3_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# ── Plot 4: Radar / spider chart ─────────────────────────────
def plot_radar(results):
    _style()
    categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]          # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#F7F7F7")

    for r in results:
        vals = [r["acc"], r["prec"], r["rec"], r["f1"]]
        vals += vals[:1]
        color = LOSS_COLORS[r["loss"]]
        ax.plot(angles, vals, linewidth=2, linestyle="solid", color=color, label=r["loss"])
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="#555")
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart — Overall Metrics", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "4_radar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# ── Plot 5: Per-class F1 heatmap ──────────────────────────────
def plot_f1_heatmap(results):
    plt.rcParams.update({"axes.grid": False, "axes.spines.top": True,
                          "axes.spines.right": True})
    loss_names = [r["loss"] for r in results]
    data = np.array([r["per_class_f1"] * 100 for r in results])   # shape (n_models, 10)

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_SHORT, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(loss_names)))
    ax.set_yticklabels(loss_names, fontsize=10)
    ax.set_title("Per-Class F1 Heatmap (%)", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(loss_names)):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    fontsize=8.5, color="black")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="F1 (%)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "5_f1_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    experiments = [
        ("CrossEntropyLoss",  "saved_model/soundclassifier_CrossEntropyLoss.pth"),
        ("BCELoss",           "saved_model/soundclassifier_BCELoss.pth"),
        ("BCEWithLogitsLoss", "saved_model/soundclassifier_BCEWithLogitsLoss.pth"),
    ]

    results  = []
    per_class = {}

    for loss_name, path in experiments:
        print(f"Evaluating: {loss_name} ...")
        r = evaluate(path, loss_name)
        if r:
            results.append(r)
            per_class[loss_name] = (r["labels"], r["preds"])

    # ── Summary comparison table ───────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'COMPARISON TABLE':^66}")
    print("=" * 70)
    print(f"  {'Loss Function':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("-" * 70)
    for r in results:
        print(f"  {r['loss']:<22} {r['acc']*100:>9.2f}% {r['prec']*100:>9.2f}% "
              f"{r['rec']*100:>9.2f}% {r['f1']*100:>9.2f}%")
    print("=" * 70)

    # ── Per-class report for each ──────────────────────────────
    for loss_name, (labels, preds) in per_class.items():
        print(f"\n\nPER-CLASS REPORT — {loss_name}:\n")
        print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    # ── Generate all plots ─────────────────────────────────────
    if results:
        print(f"\nGenerating plots → ./{PLOTS_DIR}/\n")
        plot_metric_comparison(results)
        plot_per_class_f1(results)
        plot_confusion_matrices(results)
        plot_radar(results)
        plot_f1_heatmap(results)
        print("\nAll plots saved successfully!")
