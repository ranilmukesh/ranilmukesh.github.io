---
title: "Cross-Attention with Squeeze-and-Excitation Layers for Tabular Data"
pdfUrl: "/papers/Cross-Attention-with-Squeeze-and-Excitation-Layers-for-Tabular-Data.pdf"
abstract: "This architecture is designed for tabular learning with mixed numerical and categorical features, using a shared embedding space followed by cross-attention to model interactions between feature types. It further applies squeeze-and-excitation recalibration, lightweight feedforward blocks, and efficiency techniques such as blockwise attention, sparse masking, pruning, and quantization to improve performance without making the model too expensive to train or deploy."
tags: ["Tabular Deep Learning", "Cross-Attention", "Squeeze-and-Excitation", "Feature Interaction", "Efficient AI"]
---

The model is easy to understand as a feature-interaction network: categorical fields ask questions, numerical fields provide context, and the SE layer decides which channels deserve more focus. It is a practical design for structured datasets where full transformers may be too heavy, but simple MLPs may miss important cross-feature relationships.
