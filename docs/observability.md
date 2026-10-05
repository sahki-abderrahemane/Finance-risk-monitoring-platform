# Sentinel-AI Observability

## Architecture

The observability stack provides three pillars: **metrics**, **logs**, and **traces**, plus alerting and dashboards.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  rag-engine  │────▶│  Prometheus  │────▶│  Alertmanager│
│  rl-engine   │     │              │     │              │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Grafana    │     │   Webhook    │
                     │  Dashboards  │     │   Receiver   │
                     └──────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────┐
│  OTel        │────▶│    Tempo     │
│  Collector   │     │   (traces)   │
└──────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────┐
│  Promtail    │────▶│    Loki      │
│  (logs)      │     │   (logs)     │
└──────────────┘     └──────────────┘
```

## Local Development

### Start the full stack

```bash
cd infrastructure/observability
docker compose up -d
```

This starts: Prometheus (9090), Alertmanager (9093), Grafana (3000), Loki (3100), Tempo (3200), OTel Collector (4317), Promtail, and the webhook receiver (5001).

### Start your services

```bash
# rag-engine
cd services/rag-engine
uvicorn app.main:app --port 8003

# rl-engine
cd services/rl-engine
uvicorn app.api.main:app --port 8000
```

### Access UIs

| Service       | URL                    | Credentials |
|---------------|------------------------|-------------|
| Prometheus    | http://localhost:9090   | —           |
| Alertmanager  | http://localhost:9093   | —           |
| Grafana       | http://localhost:3000   | admin/admin |
| Loki          | http://localhost:3100   | —           |
| Tempo         | http://localhost:3200   | —           |

## Endpoints

Every instrumented service exposes:

| Endpoint   | Description                           |
|------------|---------------------------------------|
| `/health`  | Liveness probe (service + deps)       |
| `/ready`   | Readiness probe (deps check)          |
| `/metrics` | Prometheus metrics scrape endpoint    |

## Metrics

All custom metrics use the `sentinel_` prefix and low-cardinality labels.

### HTTP Metrics

| Metric                                        | Type    | Labels                          |
|-----------------------------------------------|---------|---------------------------------|
| `sentinel_http_requests_total`                | Counter | service, method, endpoint, status_code |
| `sentinel_http_request_duration_seconds`      | Histogram | service, method, endpoint     |
| `sentinel_http_requests_in_progress`          | Gauge   | service, method, endpoint       |

### ML Inference

| Metric                                        | Type    | Labels                          |
|-----------------------------------------------|---------|---------------------------------|
| `sentinel_inference_requests_total`           | Counter | service, model_name, model_version, status |
| `sentinel_inference_latency_seconds`          | Histogram | service, model_name, model_version |
| `sentinel_model_loaded`                       | Gauge   | service, model_name, model_version |
| `sentinel_model_load_failures_total`          | Counter | service, model_name             |

### RAG Retrieval

| Metric                                        | Type    | Labels                          |
|-----------------------------------------------|---------|---------------------------------|
| `sentinel_retrieval_requests_total`           | Counter | service, status                 |
| `sentinel_retrieval_latency_seconds`          | Histogram | service                        |
| `sentinel_retrieved_documents_count`          | Histogram | service                        |
| `sentinel_retrieval_empty_results_total`      | Counter | service                         |

### Data Quality & Risk

| Metric                                        | Type    | Labels                          |
|-----------------------------------------------|---------|---------------------------------|
| `sentinel_risk_score_distribution`            | Histogram | service, model_name            |
| `sentinel_data_quality_failures_total`        | Counter | service, failure_type           |
| `sentinel_missing_feature_rate`               | Gauge   | service, feature_name           |

### Explanations

| Metric                                        | Type    | Labels                          |
|-----------------------------------------------|---------|---------------------------------|
| `sentinel_explanation_requests_total`         | Counter | service, status                 |
| `sentinel_explanation_latency_seconds`        | Histogram | service                        |
| `sentinel_explanation_groundedness`           | Gauge   | service                         |

## Alert Rules

Alert rules are defined in `infrastructure/kubernetes/base/observability-rules.yaml` and `infrastructure/observability/prometheus/alert-rules.yml`.

### Platform Alerts

| Alert                              | Condition                           | Severity |
|------------------------------------|-------------------------------------|----------|
| `SentinelAIPodRestarting`          | >3 restarts in 15min               | warning  |
| `SentinelAIPodNotReady`            | Unready for >10min                  | warning  |
| `SentinelAIHighCPU`                | CPU >80% for >10min                 | warning  |
| `SentinelAIHighMemory`             | Memory >85% for >10min              | warning  |

### HTTP Alerts

| Alert                              | Condition                           | Severity |
|------------------------------------|-------------------------------------|----------|
| `SentinelAIHighErrorRate`          | 5xx rate >5% for 5min              | critical |
| `SentinelAIHighLatency`            | p99 >5s for 5min                    | warning  |

### ML Alerts

| Alert                              | Condition                           | Severity |
|------------------------------------|-------------------------------------|----------|
| `SentinelAIInferenceErrors`        | Error rate >10% for 5min           | critical |
| `SentinelAIInferenceSlow`          | p99 inference >10s                  | warning  |
| `SentinelAIModelLoadFailure`       | Any load failure                    | critical |

### RAG Alerts

| Alert                              | Condition                           | Severity |
|------------------------------------|-------------------------------------|----------|
| `SentinelAIRetrievalErrors`        | Error rate >10% for 5min           | critical |
| `SentinelAIRetrievalSlow`          | p99 >5s for 5min                    | warning  |
| `SentinelAIRetrievalEmpty`         | >20% empty results for 10min       | warning  |

## Grafana Dashboards

Dashboards are auto-provisioned to Grafana on startup:

- **Sentinel-AI Platform Overview** — HTTP metrics, ML inference, RAG retrieval, data quality, risk scores, explanations
- **Sentinel-AI Kubernetes** — Pod CPU, memory, restarts, readiness, network

## Kubernetes Deployment

For Kubernetes, observability resources are in `infrastructure/kubernetes/base/`:

- `observability-prometheus-config.yaml` — Prometheus scrape configuration
- `observability-rules.yaml` — Alert rules (platform, HTTP, ML, RAG, data quality, explanations)
- `alertmanager-config.yaml` — Alertmanager routing
- `alertmanager-deployment.yaml` — Alertmanager deployment + service
- `prometheus-deployment.yaml` — Prometheus deployment
- `prometheus-service.yaml` — Prometheus service

## Environment Variables

| Variable                    | Default                          | Description                |
|-----------------------------|----------------------------------|----------------------------|
| `SENTINEL_ENVIRONMENT`      | development                      | Deployment environment     |
| `SENTINEL_METRICS_ENABLED`  | true                             | Enable Prometheus metrics  |
| `SENTINEL_LOG_LEVEL`        | INFO                             | Log level                  |
| `SENTINEL_LOG_FORMAT`       | json                             | Log format (json/console)  |
| `SENTINEL_TRACING_ENABLED`  | true                             | Enable OpenTelemetry       |
| `SENTINEL_OTLP_ENDPOINT`    | http://localhost:4317            | OTLP exporter endpoint     |
| `SENTINEL_SERVICE_NAME`     | sentinel-ai                      | Service name in traces     |
| `SENTINEL_SAMPLE_RATE`      | 1.0                              | Trace sample rate          |
