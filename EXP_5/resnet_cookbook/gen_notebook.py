"""Generate ResNet Cookbook notebook programmatically."""
import json, os

cells = []

def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]})

def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": src if isinstance(src,list) else [src]})

def img(title, path, cap=""):
    lines = []
    if title: lines.append(f"**{title}**\n")
    lines.append(f"![{cap}]({path})\n")
    md(lines)

# ===== SECTION 1: TITLE =====
md([
"# 🧠 ResNet Cookbook: Understanding Residual Learning & Skip Connections\n",
"\n",
"**Experiment 5 — Deep Learning Lab**\n",
"\n",
"---\n",
"\n",
"## 📋 Learning Objectives\n",
"\n",
"By the end of this cookbook, you will be able to:\n",
"\n",
"1. **Explain** the vanishing gradient problem and why it occurs in deep networks\n",
"2. **Derive** the residual learning formulation $y = \\mathcal{F}(x) + x$ and its gradient properties\n",
"3. **Compare** four network configurations and interpret their training dynamics\n",
"4. **Prove** — with per-layer evidence — that skip connections fix gradient flow\n",
"\n",
"## 🔬 Experiment Overview\n",
"\n",
"We train **four** model variants on the **SVHN** (Street View House Numbers) dataset:\n",
"\n",
"| # | Configuration | Residual Blocks? | Skip (`F(x)+x`)? |\n",
"|---|--------------|:-:|:-:|\n",
"| 1 | Simple CNN | ❌ | ❌ |\n",
"| 2 | Simple CNN + Skip | ❌ | ✅ |\n",
"| 3 | ResNet (no skip) | ✅ | ❌ |\n",
"| 4 | ResNet (with skip) | ✅ | ✅ |\n",
"\n",
"All models share the same depth (~20 layers), dataset, hyperparameters, and initialization."
])

# ===== SECTION 2: VANISHING GRADIENTS =====
md([
"---\n",
"# Part I: Theory\n",
"---\n",
"\n",
"## 1. The Vanishing Gradient Problem\n",
"\n",
"### Why Deep Networks Struggle\n",
"\n",
"Neural networks learn by **backpropagation**. For a network with $L$ layers, the gradient at layer $l$ is:\n",
"\n",
"$$\\frac{\\partial \\mathcal{L}}{\\partial W_l} = \\frac{\\partial \\mathcal{L}}{\\partial a_L} \\cdot \\prod_{i=l}^{L-1} \\frac{\\partial a_{i+1}}{\\partial a_i}$$\n",
"\n",
"### The Core Issue\n",
"\n",
"Each Jacobian factor $\\frac{\\partial a_{i+1}}{\\partial a_i}$ depends on the activation function and weights:\n",
"\n",
"- **Sigmoid/Tanh**: Derivatives bounded in $(0, 0.25]$ / $(0, 1]$\n",
"- **ReLU**: 0 or 1, but weight matrices can still cause shrinkage\n",
"\n",
"When many factors < 1, the product **exponentially decays**:\n",
"\n",
"$$\\prod_{i=l}^{L-1} \\frac{\\partial a_{i+1}}{\\partial a_i} \\approx \\alpha^{L-l}, \\quad |\\alpha| < 1$$\n",
"\n",
"**Result**: Early layers receive tiny gradients → tiny weight updates → they stop learning.\n",
"\n",
"### Visual Intuition\n",
"\n",
"Think of it like a game of telephone — the error signal starts strong at the output but gets **weakened** as it passes backward through each layer. By the time it reaches the first layers, it's barely a whisper.\n",
"\n",
"> **Key Insight**: The deeper the network, the worse this problem gets."
])

# ===== SECTION 3: RESIDUAL LEARNING =====
md([
"## 2. The Residual Learning Framework (He et al., 2015)\n",
"\n",
"### The Big Idea\n",
"\n",
"Instead of learning the desired mapping $\\mathcal{H}(x)$ directly, learn the **residual**:\n",
"\n",
"$$\\mathcal{F}(x) = \\mathcal{H}(x) - x \\quad \\Rightarrow \\quad y = \\mathcal{F}(x) + x$$\n",
"\n",
"- $\\mathcal{F}(x)$ = what the layers learn (the **residual**)\n",
"- $x$ = the **identity shortcut** (skip connection)\n",
"\n",
"### Why Learning Residuals is Easier\n",
"\n",
"If the optimal mapping is close to identity:\n",
"- **Without skip**: Must learn $\\mathcal{H}(x) \\approx x$ — complex through nonlinear layers\n",
"- **With skip**: Only learn $\\mathcal{F}(x) \\approx 0$ — pushing weights toward zero is trivial!\n",
"\n",
"### Residual Block Structure\n",
"\n",
"```\n",
"Input (x)\n",
"  │\n",
"  ├──────────────────┐\n",
"  │                  │ (skip / identity)\n",
"  ▼                  │\n",
"Conv → BN → ReLU    │\n",
"  │                  │\n",
"  ▼                  │\n",
"Conv → BN            │\n",
"  │                  │\n",
"  ▼                  │\n",
"  + ◄────────────────┘  ← Element-wise addition\n",
"  │\n",
"  ▼\n",
"ReLU → Output (y = F(x) + x)\n",
"```\n",
"\n",
"When dimensions differ, a **1×1 conv projection shortcut** is used: $y = \\mathcal{F}(x) + W_s \\cdot x$"
])

# ===== SECTION 4: WHY SKIP FIXES GRADIENT FLOW =====
md([
"## 3. Why Skip Connections Fix Gradient Flow\n",
"\n",
"### The Mathematical Proof\n",
"\n",
"For $y = \\mathcal{F}(x) + x$, the backward gradient is:\n",
"\n",
"$$\\frac{\\partial y}{\\partial x} = \\frac{\\partial \\mathcal{F}(x)}{\\partial x} + \\mathbf{I}$$\n",
"\n",
"### Why This Changes Everything\n",
"\n",
"**Plain network** chain rule:\n",
"$$\\frac{\\partial \\mathcal{L}}{\\partial x_l} = \\frac{\\partial \\mathcal{L}}{\\partial x_L} \\cdot \\prod_{i=l}^{L-1} \\frac{\\partial \\mathcal{F}_i}{\\partial x_i} \\quad \\text{→ can vanish}$$\n",
"\n",
"**Residual network** chain rule:\n",
"$$\\frac{\\partial \\mathcal{L}}{\\partial x_l} = \\frac{\\partial \\mathcal{L}}{\\partial x_L} \\cdot \\prod_{i=l}^{L-1} \\left(\\frac{\\partial \\mathcal{F}_i}{\\partial x_i} + \\mathbf{I}\\right) \\quad \\text{→ stays healthy}$$\n",
"\n",
"The $+\\mathbf{I}$ ensures **even if $\\frac{\\partial \\mathcal{F}_i}{\\partial x_i}$ is small, the gradient still flows.**\n",
"\n",
"### Comparison Table\n",
"\n",
"| Aspect | Plain Network | Residual Network |\n",
"|--------|:-:|:-:|\n",
"| Gradient path | Single road through layers | Highway + local roads |\n",
"| If one layer blocks | All upstream layers starve | Highway keeps flowing |\n",
"| Early layer updates | Exponentially weaker | Maintained by identity path |\n",
"| Gradient magnitude | $O(\\alpha^L)$, can vanish | $O(1)$, stays healthy |\n",
"\n",
"> **Key Takeaway**: Skip connections provide a **mathematical guarantee** that gradients can flow to any layer."
])

# ===== SECTION 5: EXPERIMENT SETUP =====
md([
"---\n",
"# Part II: Experiments\n",
"---\n",
"\n",
"## 4. Experiment Setup\n",
"\n",
"### Dataset: SVHN (Street View House Numbers)\n",
"- **Task**: Classify 32×32 RGB images of digits (0–9)\n",
"- **Training**: 73,257 images | **Test**: 26,032 images\n",
"\n",
"### Common Hyperparameters\n",
"\n",
"| Parameter | Value |\n",
"|-----------|-------|\n",
"| Batch Size | 128 |\n",
"| Epochs | 20 |\n",
"| Optimizer | Adam (lr=1e-3) |\n",
"| Scheduler | StepLR (step=10, γ=0.5) |\n",
"| Loss | CrossEntropyLoss |\n",
"| Init | Kaiming Normal |\n",
"\n",
"### Metrics Tracked Per Conv Layer\n",
"\n",
"| Metric | Tells Us |\n",
"|--------|----------|\n",
"| **Gradient Norm (L2)** | How strong is the learning signal? |\n",
"| **Weight Norm (L2)** | Is capacity being utilized? |\n",
"| **Weight Delta (L2)** | How much did weights change this epoch? |\n",
"| **Gradient Ratio** | First/last layer grad — is flow balanced? |\n",
"\n",
"> Low gradient norm + low weight delta = layer is **not learning** = vanishing gradients."
])

# ===== CASE 1: SIMPLE CNN =====
md([
"---\n",
"## 5. Case 1: Simple CNN — No Residual, No Skip\n",
"\n",
"### Architecture\n",
"Plain stack of `Conv → BN → ReLU`. No residual structure, no shortcuts.\n",
"\n",
"```\n",
"Input (3×32×32) → Conv stem (64ch)\n",
"→ 3× Conv-BN-ReLU (64ch, 32×32)\n",
"→ 3× Conv-BN-ReLU (128ch, 16×16)\n",
"→ 3× Conv-BN-ReLU (256ch, 8×8)\n",
"→ AvgPool → FC(256→10)\n",
"```\n",
"\n",
"### Expected: Early layers show weaker gradients and updates than later layers."
])

code([
"# Simple CNN Model (No Residual, No Skip)\n",
"import torch.nn as nn\n",
"\n",
"class ConvBnRelu(nn.Module):\n",
"    def __init__(self, in_ch, out_ch, stride=1):\n",
"        super().__init__()\n",
"        self.block = nn.Sequential(\n",
"            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),\n",
"            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))\n",
"    def forward(self, x): return self.block(x)\n",
"\n",
"class SimpleCNN(nn.Module):\n",
"    def __init__(self, num_classes=10):\n",
"        super().__init__()\n",
"        self.stem = ConvBnRelu(3, 64)\n",
"        channels = [64,64,64, 128,128,128, 256,256,256]\n",
"        layers, in_ch = [], 64\n",
"        for idx, out_ch in enumerate(channels):\n",
"            stride = 2 if idx in (3, 6) else 1\n",
"            layers.append(ConvBnRelu(in_ch, out_ch, stride=stride))\n",
"            in_ch = out_ch\n",
"        self.features = nn.Sequential(*layers)\n",
"        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))\n",
"        self.fc = nn.Linear(256, num_classes)\n",
"    def forward(self, x):\n",
"        x = self.stem(x); x = self.features(x)\n",
"        return self.fc(torch.flatten(self.avgpool(x), 1))"
])

md(["### Results: Simple CNN\n"])
c1 = "../wo_resnet_wo_skip/plots"
for t, f in [("📈 Accuracy Curves","accuracy_curves.png"),("📉 Loss Curves","loss_curves.png"),
    ("🔬 Epoch Gradient Norms","epoch_gradient_norms.png"),("🌡️ Gradient Heatmap","gradient_heatmap.png"),
    ("📊 Gradient Ratio","gradient_ratio.png"),("⚖️ Weight Norms","weight_norms.png"),
    ("📐 Weight Deltas","weight_deltas.png"),("🎯 Confusion Matrix","confusion_matrix.png")]:
    img(t, f"{c1}/{f}")

md([
"### 🔍 Observations — Simple CNN\n",
"\n",
"1. **Gradient Vanishing**: Early layer gradient norms are noticeably **lower** than later layers\n",
"2. **Weak Early Updates**: Weight deltas decrease for earlier layers\n",
"3. **Gradient Ratio < 1**: First/last gradient ratio is well below 1.0\n",
"4. **Baseline Reference**: Performance is limited by poor gradient flow to early feature extractors"
])

# ===== CASE 2: CNN + SKIP =====
md([
"---\n",
"## 6. Case 2: Simple CNN with Skip Connections\n",
"\n",
"Same plain CNN structure as Case 1, but with **skip connections** added.\n",
"Tests whether skip connections alone (without residual blocks) improve gradient flow.\n",
"\n",
"### Expected: Early layer gradients stronger than Case 1."
])

md(["### Results: CNN + Skip\n"])
c2 = "../wo_resnet_w_skip/plots"
for t, f in [("📈 Accuracy","accuracy_curves.png"),("📉 Loss","loss_curves.png"),
    ("🔬 Epoch Gradients","epoch_gradient_norms.png"),("🌡️ Gradient Heatmap","gradient_heatmap.png"),
    ("📊 Gradient Ratio","gradient_ratio.png"),("⚖️ Weight Norms","weight_norms.png"),
    ("📐 Weight Deltas","weight_deltas.png"),("🎯 Confusion Matrix","confusion_matrix.png"),
    ("🖼️ Sample Predictions","sample_predictions.png")]:
    img(t, f"{c2}/{f}")

md([
"### 🔍 Observations — CNN + Skip\n",
"\n",
"1. **Improved Gradient Flow**: Skip connections visibly improve early layer gradient norms\n",
"2. **Better Ratio**: First/last gradient ratio moves closer to 1.0\n",
"3. **Stronger Updates**: Early layers now get more meaningful weight updates\n",
"\n",
"> Skip connections **alone** are beneficial, even without residual block design."
])

# ===== CASE 3: RESNET NO SKIP =====
md([
"---\n",
"## 7. Case 3: ResNet — WITHOUT Skip Connections\n",
"\n",
"ResNet-20 architecture with `PlainBlock` (two convs per block) but **no `F(x)+x` addition**.\n",
"Isolates the question: *Is block structure alone sufficient?*\n",
"\n",
"```python\n",
"def forward(self, x):  # PlainBlock\n",
"    out = self.relu(self.bn1(self.conv1(x)))\n",
"    out = self.relu(self.bn2(self.conv2(out)))  # NO + x\n",
"    return out\n",
"```"
])

code([
"# PlainBlock: ResNet shape WITHOUT skip connection\n",
"class PlainBlock(nn.Module):\n",
"    def __init__(self, in_ch, out_ch, stride=1):\n",
"        super().__init__()\n",
"        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)\n",
"        self.bn1 = nn.BatchNorm2d(out_ch)\n",
"        self.relu = nn.ReLU(inplace=True)\n",
"        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)\n",
"        self.bn2 = nn.BatchNorm2d(out_ch)\n",
"        # NO skip connection\n",
"    def forward(self, x):\n",
"        out = self.relu(self.bn1(self.conv1(x)))\n",
"        out = self.relu(self.bn2(self.conv2(out)))  # ← No + x!\n",
"        return out"
])

md(["### Results: ResNet without Skip\n"])
c3 = "../outputs_resnet_wo_skip/plots"
for t, f in [("📈 Accuracy","accuracy_curves.png"),("📉 Loss","loss_curves.png"),
    ("🔬 Epoch Gradients","epoch_gradient_norms.png"),("🌡️ Gradient Heatmap","gradient_heatmap.png"),
    ("📊 Gradient Ratio","gradient_ratio.png"),("⚖️ Weight Norms","weight_norms.png"),
    ("📐 Weight Deltas","weight_deltas.png"),("🔢 Batch Gradient Norms","batch_gradient_norms.png"),
    ("🎯 Confusion Matrix","confusion_matrix.png"),("🖼️ Predictions","sample_predictions.png"),
    ("🖼️ SVHN Samples","sample_images.png")]:
    img(t, f"{c3}/{f}")

# Tables
c3t = "../outputs_resnet_wo_skip/tables"
for t, f in [("📋 Architecture","table1_architecture.png"),("📋 Performance","table2_performance.png"),
    ("📋 Training Metrics","table3_training_metrics.png"),("📋 Gradient Summary","table4_gradient_summary.png")]:
    img(t, f"{c3t}/{f}")

md([
"### 🔍 Observations — ResNet (no skip)\n",
"\n",
"1. **Gradient Vanishing Persists**: Removing skip addition causes same vanishing pattern\n",
"2. **Block Structure ≠ Solution**: Two-conv-per-block alone does NOT fix gradient flow\n",
"3. **Early Layer Starvation**: Gradient heatmap shows clear weakening in early layers\n",
"\n",
"> **Critical**: Removing just `+ x` from ResNet blocks degrades gradient flow dramatically."
])

# ===== CASE 4: RESNET WITH SKIP =====
md([
"---\n",
"## 8. Case 4: ResNet WITH Skip Connections ✅\n",
"\n",
"Complete ResNet-20 with **skip connections**: $y = \\mathcal{F}(x) + x$\n",
"\n",
"```python\n",
"def forward(self, x):  # BasicBlock\n",
"    identity = self.skip(x)                   # Skip path\n",
"    out = self.relu(self.bn1(self.conv1(x)))\n",
"    out = self.bn2(self.conv2(out))\n",
"    out += identity                            # ✅ F(x) + x\n",
"    return self.relu(out)\n",
"```\n",
"\n",
"**Architecture**: Initial Conv + 3 stages × 3 BasicBlocks + AvgPool + FC = **20 layers**"
])

code([
"# BasicBlock: ResNet WITH skip connection\n",
"class BasicBlock(nn.Module):\n",
"    def __init__(self, in_ch, out_ch, stride=1):\n",
"        super().__init__()\n",
"        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)\n",
"        self.bn1 = nn.BatchNorm2d(out_ch)\n",
"        self.relu = nn.ReLU(inplace=True)\n",
"        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)\n",
"        self.bn2 = nn.BatchNorm2d(out_ch)\n",
"        self.skip = nn.Sequential()\n",
"        if stride != 1 or in_ch != out_ch:\n",
"            self.skip = nn.Sequential(\n",
"                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),\n",
"                nn.BatchNorm2d(out_ch))\n",
"    def forward(self, x):\n",
"        identity = self.skip(x)\n",
"        out = self.relu(self.bn1(self.conv1(x)))\n",
"        out = self.bn2(self.conv2(out))\n",
"        out += identity  # ✅ SKIP CONNECTION\n",
"        return self.relu(out)"
])

md(["### Results: ResNet WITH Skip\n"])
c4 = "../outputs_resnet_with_skip/plots"
for t, f in [("📈 Accuracy","accuracy_curves.png"),("📉 Loss","loss_curves.png"),
    ("🔬 Epoch Gradients","epoch_gradient_norms.png"),("🌡️ Gradient Heatmap","gradient_heatmap.png"),
    ("📊 Gradient Ratio","gradient_ratio.png"),("⚖️ Weight Norms","weight_norms.png"),
    ("📐 Weight Deltas","weight_deltas.png"),("🔢 Batch Gradients","batch_gradient_norms.png"),
    ("🎯 Confusion Matrix","confusion_matrix.png"),("🖼️ Predictions","sample_predictions.png"),
    ("🖼️ SVHN Samples","sample_images.png")]:
    img(t, f"{c4}/{f}")

md([
"### 🔍 Observations — ResNet with Skip ✅\n",
"\n",
"1. **Healthy Gradient Flow**: Gradient norms are **consistent across all layers**\n",
"2. **Uniform Updates**: All layers receive meaningful weight updates, including earliest ones\n",
"3. **Gradient Ratio ≈ 1**: First/last gradient ratio stays close to 1.0\n",
"4. **Best Accuracy**: Highest test accuracy among all four configurations\n",
"5. **Uniform Heatmap**: No stark contrast between early and late layers\n",
"\n",
"> **Definitive proof**: `+ x` eliminates vanishing gradients and enables effective training."
])

# ===== COMPARISON =====
md([
"---\n",
"# Part III: Comparative Analysis\n",
"---\n",
"\n",
"## 9. Summary Comparison\n",
"\n",
"| Metric | Simple CNN | CNN+Skip | ResNet(no skip) | ResNet(skip) |\n",
"|--------|:-:|:-:|:-:|:-:|\n",
"| Early layer gradient | Weak ⚠️ | Improved ✅ | Weak ⚠️ | Strong ✅ |\n",
"| Gradient ratio | ≪ 1 ❌ | ~1 | ≪ 1 ❌ | ≈ 1 ✅ |\n",
"| Early weight updates | Small | Moderate | Small | Large ✅ |\n",
"| Vanishing gradients? | Yes ❌ | Reduced | Yes ❌ | No ✅ |\n",
"| Relative accuracy | Baseline | Better | ~Baseline | Best ✅ |\n",
"\n",
"### What This Proves\n",
"\n",
"1. **Depth alone causes problems** (Case 1)\n",
"2. **Skip connections help any architecture** (Case 2)\n",
"3. **Block design alone is insufficient** (Case 3)\n",
"4. **Residual learning is the complete solution** (Case 4)"
])

# ===== CONCLUSION =====
md([
"---\n",
"## 10. Key Takeaways & Conclusion\n",
"\n",
"### Core Message\n",
"\n",
"**ResNet's innovation is the skip connection**, not block structure. $y = F(x) + x$ creates a gradient highway that:\n",
"1. Prevents gradient vanishing via the identity Jacobian term\n",
"2. Makes learning identity mappings trivial ($F(x) \\to 0$)\n",
"3. Enables 100+ layer networks (ResNet-152, etc.)\n",
"\n",
"### Quick Reference\n",
"\n",
"| Concept | Formula |\n",
"|---------|--------|\n",
"| Plain mapping | $y = \\mathcal{F}(x)$ |\n",
"| Residual mapping | $y = \\mathcal{F}(x) + x$ |\n",
"| Plain gradient | $\\frac{\\partial y}{\\partial x} = \\frac{\\partial \\mathcal{F}}{\\partial x}$ |\n",
"| Residual gradient | $\\frac{\\partial y}{\\partial x} = \\frac{\\partial \\mathcal{F}}{\\partial x} + \\mathbf{I}$ |\n",
"\n",
"### Viva Checklist ✅\n",
"\n",
"1. Start with vanishing gradient chain-rule math\n",
"2. Define $y=F(x)$ vs $y=F(x)+x$\n",
"3. Explain why $+\\mathbf{I}$ creates gradient highway\n",
"4. Show per-layer gradient norms, heatmaps, ratios\n",
"5. Conclude with first-layer vs last-layer evidence\n",
"\n",
"> 💡 Strongest argument = **per-layer training dynamics**, not just top-line accuracy.\n",
"\n",
"### References\n",
"- He et al. (2015). *Deep Residual Learning for Image Recognition*. CVPR 2016.\n",
"- SVHN: Netzer et al. *Reading Digits in Natural Images*. NIPS Workshop 2011."
])

# ===== BUILD NOTEBOOK =====
nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.10.0"}},"nbformat":4,"nbformat_minor":5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ResNet_Cookbook.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {out}")
print(f"Total cells: {len(cells)} | Markdown: {sum(1 for c in cells if c['cell_type']=='markdown')} | Code: {sum(1 for c in cells if c['cell_type']=='code')}")
