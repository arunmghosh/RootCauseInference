# Downstream Inference Performance (On Full Rule-Based Reconstructed Graphs)

| Model | Quality MAE | Quality MSE | F1-Score | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GNN (Proposed) | 0.0415 | 0.003510 | 0.2582 | 0.2038 | 0.3522 | 0.7913 | 0.6645 | 0.2289 |
| MLP Baseline | 0.0481 | 0.004231 | 0.2251 | 0.1298 | 0.8457 | 0.3997 | 0.6380 | 0.1539 |
| Baseline A (Naive Freq) | 0.0644 | 0.008437 | 0.2252 | 0.1402 | 0.5717 | 0.5943 | 0.5942 | 0.2494 |
| Baseline B (Failure-Only Freq) | 0.0306 | 0.003492 | 0.2266 | 0.1530 | 0.4370 | 0.6924 | 0.5812 | 0.1372 |
| Baseline C (Constraint Elimination) | 0.0682 | 0.008268 | 0.2025 | 0.1237 | 0.5565 | 0.5479 | 0.5517 | 0.3630 |
| Baseline D (Bayesian Inference) | 0.0332 | 0.003273 | 0.2431 | 0.2014 | 0.3065 | 0.8032 | 0.6339 | 0.1642 |
