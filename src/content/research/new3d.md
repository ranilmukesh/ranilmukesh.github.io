---
title: "BrepHCC: Lightweight Cross-Modal B-Rep Point Fusion with Hierarchical Contrastive Clustering for Non-Categorical 3D Jewelry CAD Organization"
pdfUrl: "/papers/new3d.pdf"
abstract: "BrepHCC is an unsupervised 3D CAD clustering framework built for large jewelry design repositories, where visual similarity alone is not enough to capture style, topology, and manufacturing structure. It fuses native Rhino B-Rep topology with dense point-cloud features through a lightweight top-2 mixture-of-experts router and trains with a hierarchical contrastive clustering loss that preserves geometric detail, curvature patterns, and cross-model similarity."
tags: ["3D CAD", "B-Rep", "Point Clouds", "Clustering", "Jewelry Design"]
---

This work is valuable because it keeps the original CAD topology instead of throwing it away during point sampling, which means the model can reason about exact surfaces, edges, and adjacency relationships. For jewelry catalogs, that is useful for grouping similar designs, finding near-duplicates, and organizing massive archives without manual labeling.
