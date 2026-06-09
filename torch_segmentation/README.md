# Torch reimplementation of the segmentation step

What changed from the original (keras) pipeline.

- **More augmentation.** We added synced flips, rotation, and brightness/contrast jitter (plus a stronger variant with zoom, hue/sat shifts, and blur). Everything is applied to both the image and the mask together. The original built an augmented dataset but never actually trained on it, so it trained with none.

- **We track snake IoU now.** The headline metric is foreground-only IoU (the snake), not the 2-class mean the original reported. That old mean averaged in the near-perfect background, so it looked high while hiding how good the snake segmentation actually was.

- **We also track leakage.** Leakage is the fraction of predicted-snake pixels that are really background. This is the one that matters most for us, because leaked background pixels poison the downstream color clustering

- **Save the best checkpoint, not the last.** We keep the epoch with the best validation snake IoU. The original just saved the final epoch, which was slightly worse than its own best.

- **Frozen BatchNorm fix.** In the frozen-encoder stage the batchnorm layers were still drifting their running statistics, so the encoder was not really frozen. We force frozen layers' batchnorm into eval mode so it stays put

- **Stage 2 dropped.** Full fine-tuning in the second stage gave us only minimal gains, so we just run stage 1. It performs essentially as well and is simpler and cheaper

- **U-Net++ instead of plain U-Net.** U-Net++ swaps the plain skip connections for nested, dense skip pathways (extra conv blocks bridging the encoder and decoder). This narrows the gap between the fused feature maps and localizes boundaries better, which matters for us since clean boundaries are what keep leakage down. performance increased across seeds, and it's barely heavier.

- **Encoder pretrained with noisy-student, not plain ImageNet.** The EfficientNet encoder is initialized from noisy-student weights, a semi-supervised self-training scheme that gives more robust features. We chose it because in a paired test (only the encoder init differs, everything else identical) it improved snake IoU on all 5 seeds, on average about +0.011.

- **Letterbox padding instead of squashing.** The original resized straight to 512x512, which distorts non-square photos, and that same distortion fed into the color analysis. We pad to a square first so the aspect ratio is preserved (this shouldn't matter too much anyways).

- **Excluded 5 unlabeled images.** Five images contain a snake but have no annotation, we need to fix it
