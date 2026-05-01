# FraudGuard MLOps Platform

FraudGuard is a local-first production-grade MLOps platform for real-time fraud detection.

## Goal

Build an end-to-end MLOps system using:

- IEEE-CIS Fraud Detection dataset
- Python
- FastAPI
- LightGBM
- MLflow
- Feast
- Redpanda/Kafka
- Docker
- kind Kubernetes
- Prometheus/Grafana
- Kubeflow Pipelines
- Argo CD
- Argo Rollouts
- GitHub Actions
- Security scans and reports

## Current Milestone

M1 — Repository Bootstrap + First CI

## Local Setup

```bash
make install
make lint
make test