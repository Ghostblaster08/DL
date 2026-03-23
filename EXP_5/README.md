# Experiment 5: Vanishing Gradients, ResNet, and Skip Connections

## Goal

Show, with per-layer evidence, how gradient flow and weight updates change across three model families on SVHN:

1. Simple CNN (no residual blocks, no skip)
2. ResNet-style network without skip addition
3. ResNet-style network with skip addition (identity mapping and F(x)+x)

The concept and implementation are the same as your existing experiment; this update fixes naming and adds a notebook cookbook workflow.

## Correct Naming Convention

- Old phrase: Without ResNet
- Correct phrase: Simple CNN baseline (no residual blocks)

- Old phrase: With ResNet, Without Skip
- Correct phrase: ResNet-style without skip addition

- Old phrase: With ResNet, With Skip
- Correct phrase: ResNet-style with skip connections

## Existing Scripts in EXP_5

- resnet_wo_skip.py: ResNet-style network without skip addition
- resnet_with_skip.py: ResNet-style network with skip connections
- wo_resnet_w_skip.ipynb: notebook implementation of ResNet with skip
- wo_resnet_wo_skip/resnet_wo_skip.ipynb: simple CNN baseline notebook

## New Notebook Cookbook

A cookbook folder is available:

- resnet_cookbook/

Run full notebooks in this order:

1. 10_simple_cnn_full.ipynb
2. 20_resnet_no_skip_full.ipynb
3. 30_resnet_with_skip_full.ipynb
4. 40_tables_and_architecture.ipynb
5. 50_compare_all_full.ipynb

Each full notebook is self-contained and includes:

- model definitions used in that experiment
- SVHN loading and augmentation
- per-convolution-layer tracking
- training, evaluation, and artifact generation

## What Is Tracked for Every Conv Layer

- Gradient norm (L2), batch-wise and epoch-wise
- Weight norm (L2), epoch-wise
- Weight delta (L2), epoch-wise

This directly demonstrates where backpropagation updates weaken and how identity mapping helps preserve update flow.

## Outputs

Each cookbook training notebook saves under resnet_cookbook/outputs/<experiment_name>/:

- training_history.json
- models/<name>_best.pth
- models/<name>_final.pth
- models/<name>.onnx
- models/_<name>.onnx.svg
- plots/accuracy_curves.png
- plots/loss_curves.png
- plots/epoch_gradient_norms.png
- plots/batch_gradient_norms.png
- plots/gradient_heatmap.png
- plots/gradient_ratio.png
- plots/weight_norms.png
- plots/weight_deltas.png
- plots/confusion_matrix.png
- plots/sample_images.png
- plots/sample_predictions.png
- plots/table1_architecture.png
- plots/table2_performance.png
- plots/table3_training_metrics.png
- plots/table4_gradient_summary.png

## How to Run

From EXP_5:

```bash
pip install -r requirements.txt
```

Open and run the full notebooks in resnet_cookbook/ in sequence.

For script versions:

```bash
python resnet_wo_skip.py
python resnet_with_skip.py
```

No standalone cookbook Python scripts are required; the cookbook is notebook-only.
