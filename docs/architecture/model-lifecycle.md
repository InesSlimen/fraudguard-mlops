# FraudGuard Model Lifecycle

## Registered Model

Model name:

```text
fraudguard-risk-model.
```
Model version:

```text1.
```
## Lifecycle States

candidate → champion → archived

## Candidate

A candidate model is a newly registered model version produced from a successful training run.

## Champion

A champion model is the active approved model version.

Promotion Rules

A candidate may become champion only if:

roc_auc >= configured threshold
pr_auc >= configured threshold
recall >= configured threshold