import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os

# ============================================================
# LOAD TRAINING HISTORY
# ============================================================
with open("training_history.json", "r") as f:
    history = json.load(f)

OUTPUT_DIR = "outputs_resnet_wo_skip/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TABLE 1: ARCHITECTURE TABLE (like ResNet paper Table 1)
# ============================================================
def plot_architecture_table():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    title = "Table 1. ResNet WITHOUT Skip Connections — Architecture for SVHN (32×32 input)"

    columns = ["Layer Name", "Output Size", "PlainNet-20 (Ours)", "Operation"]
    rows = [
        ["conv1",       "32×32",  "3×3, 64, stride 1",                          "Conv + BN + ReLU"],
        ["stage1_b1",   "32×32",  "[3×3, 64] × 1\n[3×3, 64]",                   "PlainBlock (no skip)"],
        ["stage1_b2",   "32×32",  "[3×3, 64] × 1\n[3×3, 64]",                   "PlainBlock (no skip)"],
        ["stage1_b3",   "32×32",  "[3×3, 64] × 1\n[3×3, 64]",                   "PlainBlock (no skip)"],
        ["stage2_b1",   "16×16",  "[3×3, 128, stride 2] × 1\n[3×3, 128]",       "PlainBlock (no skip)"],
        ["stage2_b2",   "16×16",  "[3×3, 128] × 1\n[3×3, 128]",                 "PlainBlock (no skip)"],
        ["stage2_b3",   "16×16",  "[3×3, 128] × 1\n[3×3, 128]",                 "PlainBlock (no skip)"],
        ["stage3_b1",   "8×8",    "[3×3, 256, stride 2] × 1\n[3×3, 256]",       "PlainBlock (no skip)"],
        ["stage3_b2",   "8×8",    "[3×3, 256] × 1\n[3×3, 256]",                 "PlainBlock (no skip)"],
        ["stage3_b3",   "8×8",    "[3×3, 256] × 1\n[3×3, 256]",                 "PlainBlock (no skip)"],
        ["avg pool",    "1×1",    "Global Average Pooling",                      "AdaptiveAvgPool2d"],
        ["fc",          "10",     "256 → 10",                                    "Linear + Softmax"],
    ]

    # Color scheme
    header_color   = "#2c3e50"
    stage1_color   = "#d6eaf8"
    stage2_color   = "#d5f5e3"
    stage3_color   = "#fdebd0"
    other_color    = "#f2f3f4"
    row_colors = [
        other_color,
        stage1_color, stage1_color, stage1_color,
        stage2_color, stage2_color, stage2_color,
        stage3_color, stage3_color, stage3_color,
        other_color,
        other_color,
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        cellColours=[[c, c, c, c] for c in row_colors]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for j in range(len(columns)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)

    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=stage1_color, edgecolor="gray", label="Stage 1 — 64 channels, 32×32"),
        mpatches.Patch(facecolor=stage2_color, edgecolor="gray", label="Stage 2 — 128 channels, 16×16"),
        mpatches.Patch(facecolor=stage3_color, edgecolor="gray", label="Stage 3 — 256 channels,   8×8"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    ax.text(0.99, 0.01,
            "Total Parameters: 4,286,026   |   Trainable: 4,286,026\n"
            "FLOPs (approx): ~280M   |   Input: 3×32×32   |   No residual shortcuts",
            transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table1_architecture.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# TABLE 2: PERFORMANCE TABLE (like ResNet paper Table 3)
# ============================================================
def plot_performance_table():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    title = ("Table 2. Classification Results on SVHN Dataset\n"
             "(★ = current experiment  |  † = literature reference values)")

    columns = [
        "Model", "Depth", "Skip\nConnections",
        "Params", "Train Acc (%)", "Test Acc (%)",
        "Test Error (%)", "Notes"
    ]

    best_test   = max(history["test_acc"])
    final_train = history["train_acc"][-1]
    final_test  = history["test_acc"][-1]

    rows = [
        ["Plain CNN (no skip)†",   "20",   "✗", "~2.7M",  "~92.0",  "~88.0",  "~12.0",  "Literature ref"],
        ["VGG-style†",             "16",   "✗", "~15M",   "~95.0",  "~93.5",  "~6.5",   "Literature ref"],
        ["ResNet-20†",             "20",   "✓", "~0.27M", "~96.0",  "~95.5",  "~4.5",   "He et al. 2016"],
        ["ResNet-44†",             "44",   "✓", "~0.66M", "~97.0",  "~95.9",  "~4.1",   "He et al. 2016"],
        ["PlainNet-20 (Ours) ★",  "20",   "✗", "4.29M",
         f"{final_train:.2f}",
         f"{best_test:.2f}",
         f"{100 - best_test:.2f}",
         "SVHN, 20 epochs"],
    ]

    header_color = "#2c3e50"
    ref_color    = "#f2f3f4"
    ours_color   = "#fde8d8"   # orange tint to differentiate from skip version

    row_colors = [
        [ref_color]*8,
        [ref_color]*8,
        [ref_color]*8,
        [ref_color]*8,
        [ours_color]*8,
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        cellColours=row_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)

    for j in range(len(columns)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)

    for j in range(len(columns)):
        table[5, j].set_text_props(fontweight="bold")

    for i in range(1, len(rows) + 1):
        table[i, 5].set_text_props(fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=ours_color, edgecolor="gray", label="★ Our Experiment (ResNet WITHOUT Skip)"),
        mpatches.Patch(facecolor=ref_color,  edgecolor="gray", label="† Literature Reference Values"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2,
              fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table2_performance.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# TABLE 3: EPOCH-BY-EPOCH TRAINING METRICS TABLE
# ============================================================
def plot_training_metrics_table():
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis("off")

    title = "Table 3. Epoch-by-Epoch Training Metrics — ResNet WITHOUT Skip Connections"

    epochs = list(range(1, 21))
    train_loss  = history["train_loss"]
    train_acc   = history["train_acc"]
    test_loss   = history["test_loss"]
    test_acc    = history["test_acc"]

    first_grad  = history["epoch_grad_norms"]["initial.0"]
    last_grad   = history["epoch_grad_norms"]["stage3_block3.conv2"]
    ratio       = [f / (l + 1e-10) for f, l in zip(first_grad, last_grad)]

    first_delta = history["weight_deltas"]["initial.0"]
    last_delta  = history["weight_deltas"]["stage3_block3.conv2"]

    columns = [
        "Epoch",
        "Train Loss", "Train Acc (%)",
        "Test Loss",  "Test Acc (%)",
        "Grad Norm\n(initial.0)",
        "Grad Norm\n(stage3 last)",
        "Grad Ratio\n(first/last)",
        "Weight Δ\n(initial.0)",
        "Weight Δ\n(stage3 last)",
    ]

    rows = []
    for i in range(20):
        rows.append([
            str(epochs[i]),
            f"{train_loss[i]:.4f}",
            f"{train_acc[i]:.2f}",
            f"{test_loss[i]:.4f}",
            f"{test_acc[i]:.2f}",
            f"{first_grad[i]:.5f}",
            f"{last_grad[i]:.5f}",
            f"{ratio[i]:.2f}",
            f"{first_delta[i]:.4f}",
            f"{last_delta[i]:.4f}",
        ])

    test_acc_arr = np.array(test_acc)
    best_epoch   = int(np.argmax(test_acc_arr))

    def row_color(i):
        if i == best_epoch:
            return ["#a9dfbf"] * len(columns)
        elif test_acc[i] >= np.percentile(test_acc_arr, 75):
            return ["#d5f5e3"] * len(columns)
        elif test_acc[i] <= np.percentile(test_acc_arr, 25):
            return ["#fce4e4"] * len(columns)
        else:
            return ["#f9f9f9"] * len(columns)

    cell_colors = [row_color(i) for i in range(20)]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.8)

    header_color = "#2c3e50"
    for j in range(len(columns)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9.5)

    for j in range(len(columns)):
        table[best_epoch + 1, j].set_text_props(fontweight="bold")

    for i in range(1, 21):
        table[i, 4].set_text_props(fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor="#a9dfbf", edgecolor="gray",
                       label=f"★ Best Epoch (Epoch {best_epoch+1}, Test Acc: {test_acc[best_epoch]:.2f}%)"),
        mpatches.Patch(facecolor="#d5f5e3", edgecolor="gray",
                       label="Top 25% epochs by test accuracy"),
        mpatches.Patch(facecolor="#fce4e4", edgecolor="gray",
                       label="Bottom 25% epochs by test accuracy"),
        mpatches.Patch(facecolor="#f9f9f9", edgecolor="gray",
                       label="Middle 50% epochs"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2,
              fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    mean_ratio = np.mean(ratio)
    ax.text(0.99, 0.995,
            f"Key Insight (No Skip Connections):\n"
            f"Mean Grad Ratio (first/last): {mean_ratio:.2f}\n"
            f"→ Ratio < 0.1 may indicate vanishing gradients\n"
            f"→ Early layers may receive diminished weight updates",
            transform=ax.transAxes, fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table3_training_metrics.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# TABLE 4: GRADIENT NORM SUMMARY TABLE (all layers × key epochs)
# ============================================================
def plot_gradient_summary_table():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    title = ("Table 4. Gradient Norm Summary (L2) — Key Epochs\n"
             "ResNet WITHOUT Skip Connections  |  "
             "Diminishing norms in early layers = vanishing gradient signal")

    layer_names = list(history["epoch_grad_norms"].keys())
    key_epochs  = [0, 4, 9, 14, 19]
    epoch_labels = [f"Epoch {e+1}" for e in key_epochs]

    columns = ["Layer (Early → Late)"] + epoch_labels + ["Trend"]

    rows = []
    for layer in layer_names:
        norms = history["epoch_grad_norms"][layer]
        row   = [layer]
        for e in key_epochs:
            row.append(f"{norms[e]:.5f}")
        trend = "↓ Decreasing" if norms[-1] < norms[0] else "↑ Increasing"
        row.append(trend)
        rows.append(row)

    col_values = []
    for col_i, e in enumerate(key_epochs):
        col_vals = [history["epoch_grad_norms"][l][e] for l in layer_names]
        col_values.append((min(col_vals), max(col_vals)))

    cell_colors = []
    for row_i, layer in enumerate(layer_names):
        row_colors = ["#f2f3f4"]
        for col_i, e in enumerate(key_epochs):
            val = history["epoch_grad_norms"][layer][e]
            cmin, cmax = col_values[col_i]
            norm_val = (val - cmin) / (cmax - cmin + 1e-10)
            r = 0.85 - 0.45 * norm_val
            g = 0.90 - 0.35 * norm_val
            b = 1.00
            row_colors.append((r, g, b))
        norms = history["epoch_grad_norms"][layer]
        row_colors.append("#fce4e4" if norms[-1] < norms[0] else "#d5f5e3")
        cell_colors.append(row_colors)

    header_color = "#2c3e50"

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.4)

    for j in range(len(columns)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)

    for i in range(1, len(layer_names) + 1):
        table[i, 0].set_text_props(fontweight="bold", fontsize=9)

    legend_elements = [
        mpatches.Patch(facecolor=(0.4, 0.55, 1.0), edgecolor="gray",
                       label="Higher gradient norm (darker blue)"),
        mpatches.Patch(facecolor=(0.85, 0.90, 1.0), edgecolor="gray",
                       label="Lower gradient norm (lighter blue)"),
        mpatches.Patch(facecolor="#d5f5e3", edgecolor="gray",
                       label="Gradient norm increasing over training"),
        mpatches.Patch(facecolor="#fce4e4", edgecolor="gray",
                       label="Gradient norm decreasing over training"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2,
              fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table4_gradient_summary.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# ARCHITECTURE SVG (saved to models/)
# ============================================================
def plot_architecture_svg():
    """Draw a clean block-diagram of PlainNet-20 and save as SVG."""
    fig, ax = plt.subplots(figsize=(6, 18))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_aspect("equal")

    def draw_block(y, label, sublabel, color, height=0.7):
        rect = patches.FancyBboxPatch(
            (1, y), 4, height,
            boxstyle="round,pad=0.08",
            linewidth=1.2, edgecolor="#444", facecolor=color)
        ax.add_patch(rect)
        ax.text(3, y + height / 2 + 0.08, label,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1a1a1a")
        ax.text(3, y + height / 2 - 0.18, sublabel,
                ha="center", va="center", fontsize=7.5, color="#555")

    def draw_arrow(y_from, y_to):
        ax.annotate("", xy=(3, y_to + 0.005), xytext=(3, y_from),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))

    input_y = 16.6
    ax.text(3, input_y + 0.35, "Input  3 × 32 × 32",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#aaa"))

    blocks = [
        # (y_bottom, label, sublabel, color)
        (15.4, "Conv1  +  BN  +  ReLU", "3×3, 64 ch  |  stride 1  →  32×32", "#dfe6e9"),
        # Stage 1
        (14.0, "Stage 1 — Block 1", "PlainBlock  64→64  |  32×32  (no skip)", "#d6eaf8"),
        (12.9, "Stage 1 — Block 2", "PlainBlock  64→64  |  32×32  (no skip)", "#d6eaf8"),
        (11.8, "Stage 1 — Block 3", "PlainBlock  64→64  |  32×32  (no skip)", "#d6eaf8"),
        # Stage 2
        (10.4, "Stage 2 — Block 1", "PlainBlock  64→128  |  stride 2  →  16×16  (no skip)", "#d5f5e3"),
        (9.3,  "Stage 2 — Block 2", "PlainBlock  128→128  |  16×16  (no skip)", "#d5f5e3"),
        (8.2,  "Stage 2 — Block 3", "PlainBlock  128→128  |  16×16  (no skip)", "#d5f5e3"),
        # Stage 3
        (6.8,  "Stage 3 — Block 1", "PlainBlock  128→256  |  stride 2  →  8×8   (no skip)", "#fdebd0"),
        (5.7,  "Stage 3 — Block 2", "PlainBlock  256→256  |  8×8   (no skip)", "#fdebd0"),
        (4.6,  "Stage 3 — Block 3", "PlainBlock  256→256  |  8×8   (no skip)", "#fdebd0"),
        # Head
        (3.4,  "Global Avg Pool", "AdaptiveAvgPool2d  →  1×1", "#dfe6e9"),
        (2.2,  "Fully Connected", "256  →  10 classes", "#dfe6e9"),
        (1.0,  "Softmax Output", "10-class probabilities", "#eaf4fb"),
    ]

    prev_top = input_y
    for (y, label, sublabel, color) in blocks:
        draw_arrow(prev_top, y + 0.7)
        draw_block(y, label, sublabel, color)
        prev_top = y

    ax.set_title("PlainNet-20 (ResNet w/o Skip) — SVHN Architecture\n"
                 "No residual shortcuts  |  4,286,026 parameters",
                 fontsize=10, fontweight="bold", pad=8)

    svg_path = os.path.join("models", "_resnet_wo_skip.onnx.svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {svg_path}")


# ============================================================
# RUN ALL
# ============================================================
print("=" * 60)
print("Generating Tables from Training Results (WITHOUT Skip)")
print("=" * 60)

plot_architecture_table()
plot_performance_table()
plot_training_metrics_table()
plot_gradient_summary_table()
plot_architecture_svg()

print("\nAll outputs saved.")
print("\nPlots → ", OUTPUT_DIR)
print("  table1_architecture.png   — Layer-by-layer architecture")
print("  table2_performance.png    — Accuracy comparison vs literature")
print("  table3_training_metrics.png — Full epoch-by-epoch metrics")
print("  table4_gradient_summary.png — Gradient norms across layers and key epochs")
print("\nSVG   → models/_resnet_wo_skip.onnx.svg  (architecture block diagram)")
