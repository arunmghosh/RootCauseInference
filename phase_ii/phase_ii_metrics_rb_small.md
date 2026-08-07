# Downstream Inference Performance (On Small Rule-Based Reconstructed Graphs)

| Model | Quality MAE | Quality MSE | F1-Score | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GNN (Proposed) | 0.0408 | 0.003213 | 0.2727 | 0.1875 | 0.5000 | 0.6049 | 0.6027 | 0.2943 |
| MLP Baseline | 0.0647 | 0.005620 | 0.2469 | 0.1449 | 0.8333 | 0.2469 | 0.5851 | 0.1708 |
| Baseline A (Naive Freq) | 0.1051 | 0.015212 | 0.2432 | 0.1452 | 0.7500 | 0.3086 | 0.5694 | 0.4500 |
| Baseline B (Failure-Only Freq) | 0.0685 | 0.006886 | 0.2432 | 0.1452 | 0.7500 | 0.3086 | 0.4517 | 0.1835 |
| Baseline C (Constraint Elimination) | 0.0726 | 0.008309 | 0.3051 | 0.1915 | 0.7500 | 0.4938 | 0.5996 | 0.4893 |
| Baseline D (Bayesian Inference) | 0.0414 | 0.003450 | 0.2174 | 0.1471 | 0.4167 | 0.5556 | 0.5839 | 0.2134 |
