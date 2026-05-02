---
title: "Cerebro-TabM: Contextual Evidential Retrieval & Ensembling with Bounded Rademacher Optimization for Clinical Stroke Prediction"
pdfUrl: "/papers/cerebro_tabm-flatten.pdf"
abstract: "Cerebro-TabM is a clinical stroke prediction architecture built for highly imbalanced and partially missing healthcare data, where rare positive cases and missing-not-at-random features can mislead standard tabular models. It introduces a missingness-aware tokenizer, an evidential BatchEnsemble that outputs uncertainty-aware Dirichlet evidence, and a PAC-Bayes-grounded focal loss that aims to improve calibration, fairness, and edge-deployment readiness for safety-critical decision support."
tags: ["Clinical AI", "Stroke Prediction", "Tabular Learning", "Uncertainty Estimation", "PAC-Bayes"]
---

This paper is mainly about making tabular medical prediction more reliable under real hospital constraints, especially when the disease is rare and the missing values themselves carry clinical meaning. Instead of oversampling or chasing raw accuracy alone, it focuses on calibrated uncertainty, fairness, and generalization guarantees that are more appropriate for clinical deployment.
