---
title: "Mixture-of-Depths Meets Thresholded Differential Attention: Omni-PAIFT, a Universal Sink-Free Multimodal Foundation Transformer with Gradient-Orthogonal Fusion and Active Uncertainty Routing"
pdfUrl: "/papers/omni_paift.pdf"
abstract: "Omni-PAIFT is a unified multimodal foundation architecture designed to reduce information dilution, attention sinks, gradient interference, and expensive uncertainty estimation in large-scale systems. It combines threshold-gated mixture-of-depths differential attention, cross-modal fusion, single-pass uncertainty routing, evolutionary multi-task optimization, and speculative sparse execution to improve long-context efficiency and downstream task performance with only modest additional compute."
tags: ["Multimodal AI", "Foundation Models", "Long Context", "Uncertainty Routing", "Mixture of Depths"]
---

The paper aims to make large multimodal models both smarter and more efficient by deciding which tokens, depths, and experts deserve computation instead of treating everything equally. It is especially positioned for resource-constrained settings where long contexts, multiple modalities, and uncertainty-aware reasoning all need to coexist without a huge inference penalty.
