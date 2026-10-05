"""Minimal Alertmanager webhook receiver for local development.

Receives alerts from Alertmanager and logs them to stdout.
In production this would route to PagerDuty, Slack, or a similar system.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Alertmanager Webhook Receiver")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("alertmanager-webhook")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/alerts")
async def receive_alerts(request: Request) -> JSONResponse:
    payload = await request.json()

    alerts = payload.get("alerts", [])

    for alert in alerts:
        status = alert.get("status", "unknown")
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})

        severity = labels.get("severity", "unknown")
        alertname = labels.get("alertname", "unknown")
        service = labels.get("service", "n/a")
        summary = annotations.get("summary", "")
        description = annotations.get("description", "")

        log_fn = logger.critical if status == "firing" and severity == "critical" else logger.warning

        log_fn(
            "ALERT | status=%s severity=%s alert=%s service=%s | %s | %s",
            status,
            severity,
            alertname,
            service,
            summary.strip(),
            description.strip(),
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "received": len(alerts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
