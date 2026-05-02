---
title: "Generative Causal-Curriculum: A Privacy-Preserving Federated Framework for Synthetic Hard-Example Mining in Brain Tumor Segmentation"
pdfUrl: "/papers/Brain Tumor3d 2026-flatten.pdf"
abstract: "This work proposes a federated brain tumor segmentation framework that turns model failure patterns into anonymous causal error vectors instead of sharing raw medical images. A server-side latent diffusion model then generates synthetic hard examples that specifically target those blind spots, while MO-DGPO and TreeGRPO make the optimization more stable and reduce generation overhead for practical federated training."
tags: ["Federated Learning", "Brain Tumor Segmentation", "Diffusion Models", "Causal Inference", "Medical Imaging"]
---

The core idea is to stop relying on generic augmentation and instead create synthetic tumor patches that reflect exactly where the segmentation model is failing, such as fuzzy boundaries or low-contrast regions. That makes the framework especially relevant for privacy-sensitive hospitals dealing with rare tumor sub-regions, where class imbalance and non-IID data make conventional federated learning perform poorly.
