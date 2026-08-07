# Downstream Inference Performance (On Small NLP Reconstructed Graphs)

| Model | Quality MAE | Quality MSE | F1-Score | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GNN (Proposed) | 0.0386 | 0.003152 | 0.3571 | 0.3125 | 0.4167 | 0.7778 | 0.7391 | 0.4625 |
| MLP Baseline | 0.0414 | 0.003101 | 0.3396 | 0.2195 | 0.7500 | 0.5679 | 0.6920 | 0.3006 |
| Baseline A (Naive Freq) | 0.0612 | 0.006633 | 0.3509 | 0.2222 | 0.8333 | 0.5432 | 0.7107 | 0.4234 |
| Baseline B (Failure-Only Freq) | 0.0705 | 0.009057 | 0.3333 | 0.2083 | 0.8333 | 0.5062 | 0.6449 | 0.3735 |
| Baseline C (Constraint Elimination) | 0.0438 | 0.004792 | 0.3448 | 0.2941 | 0.4167 | 0.7654 | 0.6214 | 0.3986 |
| Baseline D (Bayesian Inference) | 0.0367 | 0.003333 | 0.2759 | 0.2353 | 0.3333 | 0.7407 | 0.7005 | 0.2512 |
