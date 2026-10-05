# Finance Risk Management Platform

**Enterprise multimodal financial risk and simulation platform — tabular, textual, visual, and sequential signals fused into one auditable risk score, explained and stress-tested.**

Built as **Sentinel-AI**, this platform fuses **gradient-boosted tabular models, FinBERT market sentiment, CNN chart patterns, late-fusion unified risk scoring, RAG-grounded explanations, sim-only reinforcement learning, SHAP/LIME explainability, and full observability (Prometheus + Grafana + Loki + Tempo)** into a modular monorepo of independently deployable FastAPI engines.

It is **research and simulation only** — it never gives financial advice and never executes trades.

---

## What the Platform Does

### Risk Scoring

Tabular market features → calibrated 0–100 score.

```text
OHLCV
 ↓
Cleaning + validation
 ↓
Features (log_return, volatility, momentum, volume_z)
 ↓
PCA (95% variance) + Gradient Boosting
 ↓
Risk score = 0.5 * model_uncertainty + 0.5 * volatility
```

### Market Sentiment

Financial news → FinBERT sentiment + embeddings.

```text
News text
    ↓
Preprocessing
    ↓
ProsusAI/FinBERT (neg / neu / pos)
    ↓
Mean-pooled window sentiment + [CLS] embedding
```

### Chart Vision

Candlestick charts → CNN pattern classifier.

```text
OHLCV → candlestick PNG (AAPL, AMZN, GOOGL, MSFT, NVDA)
    ↓
CNN (3ch, 3 classes, 224px)
    ↓
Pattern class + vision embedding
```

### Unified Multimodal Fusion

All three signals → one risk band.

```text
          ┌──→ tabular embedding ──┐
Market ───┼──→ FinBERT embedding ──┼──→ Late fusion [z_tab || e_text || e_vision]
          └──→ CNN embedding ──────┘           ↓
                                      MLP (901 → 256 → 64)
                                               ↓
                                   LOW / MODERATE / HIGH / EXTREME
```

### Grounded Explanations (never predictions)

Retrieval-augmented explainer that only explains, with evidence.

```text
Risk output
    ↓
pgvector cosine top-k retrieval
    ↓
Explain-only prompt (never predict)
    ↓
Validated explanation + evidence spans
```

### Strategy Simulation

Sim-only market environment for policy research.

```text
SimulatedMarketEnvironment
    ↓
Q-Learning → Policy Gradient → PPO (clipped objective)
    ↓
Sharpe-style reward + comparison reports
```

### Trust Layer

Every score is audited before it is trusted.

```text
Score
 ├──→ SHAP / LIME local explanations
 ├──→ FGSM adversarial robustness
 ├──→ DP-SGD privacy accounting
 ├──→ PSI / KS drift detection
 └──→ Parity-gap bias audit
```

---

## Features

| Feature | Description | Service |
|---|---|---|
| Data engine | OHLCV ingestion, cleaning, validation, features, stats, PCA | `services/data_engine` |
| Tabular risk | LightGBM/GBM scorer + MLflow tracking + REST API | `services/risk-engine`, `risk-engine-api` |
| NLP sentiment | FinBERT 3-class sentiment + aggregate windows + API | `services/nlp_engine`, `nlp-engine-api` |
| Vision | Chart dataset + CNN training/comparison + API | `services/vision-engine`, `vision-engine-api` |
| Fusion | Late-fusion dataset + unified MLP + eval + API | `services/multimodal-engine`, `multimodal-engine-api` |
| RAG explainer | Chunk → embed → pgvector → grounded explain-only LLM | `services/rag-engine` |
| RL simulation | Sim market env, Q/PPO trainers, metrics, API on `:8000` | `services/rl-engine` |
| Explainability | SHAP, LIME, adversarial, DP, drift, bias reports | `services/explainability` |
| Observability | Metrics/logs/traces, alerts, Grafana dashboards | `infrastructure/observability` |
| K8s delivery | Per-engine deployments + overlays + policies | `infrastructure/kubernetes` |

---

## Architecture

```text
                          Market Data / News / Charts
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │ data_engine  │  │ nlp_engine   │  │ vision-engine│
            │ ingest→PCA   │  │ FinBERT      │  │ CNN charts   │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     ▼
                     ┌──────────────────────────────┐
                     │ multimodal-engine            │
                     │ [z_tab || e_text || e_vision]│
                     │ → LOW…EXTREME                │
                     └──────────────┬───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ risk-engine  │ │ rag-engine   │ │ rl-engine    │
            │ 0–100 score  │ │ explain-only │ │ sim PPO      │
            │ :risk-score  │ │ :8003        │ │ :8000        │
            └──────────────┘ └──────────────┘ └──────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ explainability                 │
                     │ SHAP/LIME/FGSM/DP/drift/bias │
                     └──────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ observability                │
                     │ Prometheus/Grafana/Loki/Tempo│
                     └──────────────────────────────┘
```

Each `*-engine-api` is a thin FastAPI shim over its engine (e.g. `POST /v1/risk-score`). Every service exposes `/health`, `/ready`, `/metrics`.

---

## Project Structure

```text
sentinel/
│
├── services/
│   ├── data_engine/src/data_engine/   # ingestion, loader, cleaner, validator,
│   │                                  # features, statistics, pca, schemas, config
│   ├── risk-engine/src/risk_engine/   # gradient_boosting, risk_scoring, evaluation, mlflow
│   ├── risk-engine-api/               # main, routes, schemas, dependencies, Dockerfile
│   ├── nlp_engine/src/nlp_engine/     # sentiment (FinBERT), preprocessing, aggregate, train
│   ├── nlp-engine-api/                # FastAPI shim
│   ├── vision-engine/src/vision_engine/ # chart_generator, dataset, model (CNN), train, eval
│   ├── vision-engine-api/             # FastAPI shim
│   ├── multimodal-engine/src/multimodal_engine/ # alignment, fusion_dataset, feature_fusion,
│   │                                             # unified_risk_model (901→256→64), train, eval
│   ├── multimodal-engine-api/         # main, routes, schemas
│   ├── rl-engine/app/                 # environment/, agent/ (q, pg, ppo, actor-critic),
│   │                                  # training/, evaluation/, experiment/, api/:8000
│   ├── rag-engine/app/                # ingestion/, retrieval/, llm/, explanation/,
│   │                                  # vector_store (pgvector), main :8003
│   └── explainability/src/explainability/ # shap, lime, adversarial, differential_privacy,
│                                           # drift_detection, bias_audit + reports
│
├── apps/                              # Placeholder (Next.js dashboard + gateway per report Ch.4)
├── pipelines/                         # Placeholder (no DAGs yet)
├── configs/                           # Placeholder
├── ml/
│   ├── datasets/                      # raw, processed, features, gbm, risk_scores, statistics,
│   │                                  # pca, news, multimodal, evaluation, charts/
│   ├── experiments/ / models/         # Artifacts also under mlruns/
├── infrastructure/
│   ├── docker/                        # Dockerfile.python-service, rag/rl Dockerfiles
│   ├── kubernetes/base/               # *-engine-api deployments, rag/rl, observability,
│   │                                  # prometheus, grafana, alertmanager, policies, secrets
│   │   └── overlays/{production,staging}/
│   └── observability/                 # docker-compose, prometheus, loki, tempo, otel,
│                                      # promtail, grafana provisioning, alertmanager, webhook
│
├── tests/                             # Placeholder at root; real tests co-located per service
├── docs/observability.md
├── projectreport.md                   # Full thesis (Sentinel-AI, ESTIN Béjaïa 2025-26)
├── pyproject.toml / .env.example / Makefile
└── README.md
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Tabular ML | LightGBM / Gradient Boosting, scikit-learn, PCA, pandas, numpy |
| NLP | Hugging Face Transformers, ProsusAI FinBERT, PyTorch |
| Vision | PyTorch CNN, candlestick chart pipeline, Pillow |
| Fusion | PyTorch MLP late fusion |
| RAG | pgvector, LangSmith tracing, explain-only LLM provider |
| RL | Custom sim env, Q-Learning, Policy Gradient, PPO, MLflow |
| APIs | FastAPI, Uvicorn, Pydantic v2 |
| Data | Postgres + pgvector, Redis (per report), MLflow tracking |
| Observability | Prometheus, Alertmanager, Grafana, Loki + Promtail, Tempo, OpenTelemetry |
| Delivery | Docker, Kubernetes (base + overlays), GitHub Actions |
| Testing | Pytest (per-service `tests/`) |

---

## Quick Start

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python `>=3.13,<3.14`.

### 2. Configuration

```bash
cp .env.example .env   # then edit values
```

### 3. Run tests per engine

```bash
pytest services/data_engine/tests \
       services/risk-engine/tests \
       services/nlp_engine/tests \
       services/vision-engine/tests \
       services/multimodal-engine/tests
```

(`testpaths` + `pythonpath` are pre-wired in `pyproject.toml`.)

### 4. Run an engine API

```bash
# risk scorer
python services/risk-engine-api/main.py
# POST /v1/risk-score

# RL sim API
python services/rl-engine/app/api/main.py   # :8000

# RAG explainer
python services/rag-engine/app/main.py       # :8003
```

### 5. Observability stack

```bash
cd infrastructure/observability
docker compose up -d
```

| Service | URL |
|---|---|
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Grafana (`admin`/`admin`) | http://localhost:3000 |
| Loki | http://localhost:3100 |
| Tempo | http://localhost:3200 |
| OTel collector (gRPC) | http://localhost:4317 |
| Alert webhook | http://localhost:5001 |

### 6. MLflow

Artifacts under `mlruns/` + per-service runs. See `mlflow_tracking.py` / `run_mlflow.py` in risk and multimodal engines.

---

## Configuration (`.env.example`)

| Variable | Description |
|---|---|
| `POSTGRES_HOST` / `POSTGRES_PORT` | Postgres host/port for engine + pgvector store |
| `POSTGRES_DATABASE` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | Database credentials |
| `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` / `LANGSMITH_API_KEY` | RAG tracing |

Observability extras (`docs/observability.md`):

```text
SENTINEL_ENVIRONMENT, SENTINEL_METRICS_ENABLED, SENTINEL_LOG_LEVEL,
SENTINEL_LOG_FORMAT, SENTINEL_TRACING_ENABLED, SENTINEL_OTLP_ENDPOINT,
SENTINEL_SERVICE_NAME, SENTINEL_SAMPLE_RATE
```

---

## API Endpoints

Thin FastAPI shims per engine; every service also serves `/health`, `/ready`, `/metrics`.

```text
POST /v1/risk-score        # risk-engine-api — tabular 0–100 score
POST /...                  # nlp-engine-api — sentiment + embedding
POST /...                  # vision-engine-api — chart pattern + embedding
POST /...                  # multimodal-engine-api — LOW…EXTREME band
:8000  rl-engine/app/api   # sim rollout, training, evaluation
:8003  rag-engine/app      # retrieval, ingestion, explanation (explain-only)
```

Key RAG routes in `services/rag-engine/app/api/`: `routes.py` (retrieval), `ingestion_routes.py`, `explanation_routes.py` (+ schemas). RAG is grounding-only: `EXPLANATION_SYSTEM_PROMPT` = explain, never predict.

---

## Observability

Three pillars + alerts + dashboards (`docs/observability.md`):

```text
Services ──→ Prometheus (:9090) ──→ Alertmanager (:9093) ──→ webhook (:5001)
    ├──→ Loki (:3100 via Promtail) — logs
    ├──→ Tempo (:3200 via OTel :4317) — traces
    └──→ Grafana (:3000) — Platform Overview + Kubernetes dashboards
```

`sentinel_*` metrics: http, inference, retrieval, risk/data-quality, explanations. Alerts (pod restarts, 5xx > 5%, p99 > 5s, inference errors, empty retrieval > 20%) in `observability/prometheus/alert-rules.yml` + `kubernetes/base/observability-rules.yaml`.

---

## Current Capabilities

```text
┌─────────────────────────────────────────────┐
│        Finance Risk Management Platform     │
│                 (Sentinel-AI)               │
├─────────────────────────────────────────────┤
│                                             │
│  ✓ OHLCV ingestion + validation + PCA       │
│  ✓ GBM risk scorer + MLflow                 │
│  ✓ FinBERT sentiment + aggregation          │
│  ✓ CNN chart classifier                    │
│  ✓ Late-fusion unified risk bands           │
│  ✓ RAG grounded explainer (explain-only)    │
│  ✓ Sim-only RL (Q → PPO)                    │
│  ✓ SHAP / LIME + robustness + DP + drift    │
│  ✓ FastAPI per-engine shims                 │
│  ✓ Prometheus/Grafana/Loki/Tempo            │
│  ✓ K8s base + staging/production overlays   │
│  ✓ Per-service pytest suites                │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Status

**Phases 1–5 implemented per `projectreport.md`** (tabular → multimodal → RAG → RL → trust/K8s). `apps/`, `pipelines/`, `configs/`, root `tests/`, and `Makefile` remain placeholders. See `projectreport.md` (631 lines) and `Sentinel-AI-Report-1.pdf` for acceptance criteria (MAE, paired t-test, ≥90% groundedness, simulated Sharpe, parity/PSI).

> Safety: simulation and research only. No financial advice. No live execution.

---

## License

Add your preferred license here.
