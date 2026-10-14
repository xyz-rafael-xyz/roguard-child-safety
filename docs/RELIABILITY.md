# Reliability workflows in RoGuard 6

## Local workspace and stronger optional model

```sh
pip install --upgrade roguard-child-safety
roguard-studio --reliability --open
```

The core supports Python 3.9–3.14 and has no runtime dependencies. For the adapted encoder, use Python 3.10+ (verified on 3.12):

```sh
pip install --upgrade 'roguard-child-safety[adapted]'
roguard-model fetch
roguard-studio --conversation --backend adapted --open
```

The pinned base download is about 1.2 GB. The small verified adaptation is bundled. Inference remains local. `fast`, `semantic` and `hybrid` retain their prior models; `adapted` explicitly selects the new encoder and class-conditional calibrated prediction sets. Empty or two-label sets return `INSUFFICIENT_EVIDENCE`. Calibration requires exchangeable examples; neither its alpha nor a score is an individual safety probability. The calibration sample has only 18 positives and 12 negatives per language and is synthetic. See [all experiments](V6_RELEASE.md).

The reliability page accepts up to 128 turns / 64,000 characters. It produces overlapping windows with original turn and character references, records unsupported windows, and never silently truncates the input. A positive window requests review; negative windows alone leave the overall result unresolved. Windowing can lose distant context and split words. It does not prove whole-conversation comprehension, and its aggregate is not calibrated. Prefer the ordinary conversation API for inputs within its supported context.

```sh
roguard-ops long examples/long_conversation_ro.json
roguard-ops repair-demo --language uk
```

## Permission-aware repair

`repair_proposal(ledger, proposed, candidates, requirements, purpose=..., recipient=..., operation=...)` returns the cheapest permitted field set satisfying **unchanged** declared requirements and scope. It neither grants permission nor sends anything. The host supplies alternative artifacts, coverage and costs; the planner does not generate or semantically verify a replacement summary. Derived summaries retain source restrictions. To share a truly less-sensitive alternative, the host must register and justify its own source/permission declarations; claiming a new source is not automatic declassification.

```python
from roguard import FlowLedger, repair_proposal

with FlowLedger() as ledger:
    ledger.add_source('lesson', 'Fictional learning progress')
    ledger.add_source('personal', 'Unrelated fictional private note')
    ledger.permission('lesson', 'tutoring', 'teacher', 'export', 'grant')
    proposal = repair_proposal(
        ledger, ['lesson', 'personal'],
        [{'artifact_id': 'lesson', 'covers': ['learning'], 'cost': 1},
         {'artifact_id': 'personal', 'covers': [], 'cost': 9}],
        ['learning'], purpose='tutoring', recipient='teacher', operation='export')
    assert proposal['selected'] == ['lesson']
```

Every real execution still needs current checks and host approval. A stale repair report is not a capability.

## Authenticated service

```sh
pip install --upgrade 'roguard-child-safety[gateway]'
roguard-ops init --directory ./roguard-service
roguard-gateway --state ./roguard-service/state.db --credentials ./roguard-service/credentials.json --key-file ./roguard-service/state.key
```

Initialization creates a new private directory and separate random admin/operator/reader bearer tokens for tenant `local`. It refuses an existing directory. Read your token locally from `credentials.json`; never put it in a prompt, issue or public repository. Keep the key separate from database backups. Token configuration is loaded at startup; restart after rotation. The CLI binds loopback by default. Network deployment requires a TLS reverse proxy, protected configuration/volumes, request limits and operator-owned access management. This is a bounded service, not a claim of enterprise certification or horizontal scalability.

All routes use POST, `Content-Type: application/json`, and `Authorization: Bearer TOKEN`. Browser Origin headers are rejected; use the separate loopback Studio for browser workflows. Bodies are limited to 256 KiB, upload reads to ten seconds, and active jobs to four per application instance; excess active capacity returns 429. HTTP access logging is disabled. Other proxy/exporter logs remain the host's responsibility.

| Route | Body | Minimum role |
|---|---|---|
| `/v1/analyze` | Existing conversation payload | reader |
| `/v1/analyze-long` | `language`, `turns` | reader |
| `/v1/metrics` | `{}` or `baseline` report | reader |
| `/v1/session/create` | `session` opaque ID | admin |
| `/v1/session/mutate` | `session`, `kind` (`source`, `derive`, `permission`), `args` for the ledger method | admin |
| `/v1/repair` | `session`, `request` for `repair_proposal` | reader |
| `/v1/job/enqueue` | `session`, `tool`, `artifact_ids`, optional stable `request_id` | operator |
| `/v1/job/approve` | `job` ID | admin |
| `/v1/job/cancel` | `job` ID | operator |
| `/v1/job/execute` | `job` ID | admin |
| `/v1/job/status` | `job` ID | reader |
| `/v1/session/erase` | `session` ID | admin |
| `/v1/metrics/erase` | `{}` | admin |

Tenants come from server-side credentials, never the body. The CLI has no tool bindings; applications register trusted `ToolBinding` objects using `DurableGateway` and `GatewayApp`, as in [the runnable local example](../examples/gateway_v6.py). There is no HTTP route for registering Python, destinations or arbitrary callbacks. `binding_version` must change whenever callback semantics, destinations or configuration change. Changed scope/version invalidates old jobs.

The gateway encrypts source values and policy journals with Fernet and binds encrypted session state to its tenant/session. SQLite metadata (IDs/status/timestamps) remains readable to the file owner. Final checks and synchronous callbacks run under a database write transaction. This serializes workers sharing that file and can limit throughput. Callbacks must be bounded, trusted and must not reenter this database. It is not a sandbox.

Jobs require explicit approval, expire after five minutes and are invalidated by policy/provenance changes. Permission expiry uses Unix seconds, with a persisted nondecreasing clock. `STARTED` is committed before invoking a callback. After a crash it is reported as uncertain and cannot be replayed; a crash before the callback may therefore lose an action. No exactly-once remote-delivery claim is made. A callback that throws may already have acted. Request IDs deduplicate enqueue retries; never create a new ID automatically after an uncertain action. Cancellation works before execution starts; disconnecting the HTTP caller does not cancel a running callback. Async HTTP execution uses a thread, not interruptible Python.

A remote receiver needs its **own** final authorization and idempotency boundary. SQLite does not atomically control a third-party service. Host approval authenticates the configured service principal, not a guardian's legal identity. Session limits are 100 per tenant, 2,048 events per session, and 1,000 jobs per tenant; erase finished sessions explicitly. Deletion does not guarantee cryptographic erasure from backups or filesystem pages.

## Monitoring and OpenTelemetry

```python
from roguard import Monitor

monitor = Monitor(retention_seconds=86400, max_events=10000)
monitor.record(tenant='demo', language='ro', decision='INSUFFICIENT_EVIDENCE',
               latency_ms=12, model_id='frozen_model', policy_version='v6')
assert monitor.report('demo')['languages']['ro']['count'] == 1
monitor.close()
```

Monitoring accepts only fixed metadata fields: language, decision, latency, model/policy/dataset IDs. Use opaque IDs, never names or message fragments. Reports separate Romanian/Ukrainian counts, abstention and p95 latency. Comparison with a saved baseline reports total-variation distance; an alert requires at least 30 events in both samples and distance above 0.2. This is a descriptive heuristic, not an accuracy or significance test. Model changes and population changes can both move the distribution.

`roguard-ops monitor DATABASE --tenant TENANT --baseline BASELINE.json` prints the report. Retention purging happens on record/report access; it is not a background deletion scheduler. The global row cap bounds disk records and a noisy tenant may evict older events. No raw-text incident replay is retained: the host can create an explicitly reviewed fictional reproducer in the interaction lab.

Install `roguard-child-safety[observability]` and pass a host-configured OpenTelemetry tracer as `Monitor(tracer=...)`. The standard SDK can use its OTLP exporter to a configured collector/Langfuse-compatible integration. RoGuard does not configure or contact an exporter automatically. Spans contain the six declared fields, no prompts, exception messages, credentials or source values. The span's duration covers emission; inference latency is the explicit `roguard.latency_ms` attribute. Exporter failures/availability are the host's responsibility. [OpenTelemetry setup](https://opentelemetry.io/docs/languages/python/instrumentation/).

## Calibrating another model and detecting feature shift

`CalibratedDecisions.fit(scores, labels, model_id=..., language=..., alpha=0.1)` creates a JSON artifact. Use separate calibration families; never fit against the final test. `predict` rejects a different model/task identity or language. `analyze_calibrated_conversation` applies it to a matching scorer while retaining structural/language abstentions. `FeatureSupport.fit(reference_vectors, calibration_vectors, model_id=...)` adds a diagonal-distance diagnostic when passed as `support=`. Outside-reference vectors force abstention. Distance is not a semantic novelty detector and cannot reliably identify every unfamiliar input. See [the conformal introduction](https://arxiv.org/abs/2107.07511) for assumptions.

## Actual model comparisons

`ChatTarget(endpoint, model, api_key=...)` is an explicit chat-completions adapter; HTTPS is required outside loopback and redirects are rejected. It sends observation content to the chosen endpoint, so use fictional scenarios or properly authorized content. There is no default endpoint or credential discovery. `run_scenario(..., target_factory=lambda: ChatTarget(...))` retains errors and invalid output in its metrics. Model code cannot grant permissions.

The local comparison runner `python -m eval.compare_local_targets_v6 --model NAME=CACHED_MODEL_DIRECTORY --output NEW.json` requires the `mlx` extra and an Apple GPU. It records model file hashes, two stochastic repeats and per-language results. The host minimum-disclosure proposal is visible to the model; this measures following a supplied plan and completing a task, not independently inferring policy. Both initial and revised-contract runs are retained. It is a comparison of these configurations, not a market-wide leaderboard.

## V7 context review integration

The optional `/v1/analyze-context` endpoint accepts the bounded context payload when `GatewayApp` is constructed with an explicit `context_model`. It uses the same authenticated reader role and request limits. Configure `--context-backend joint` on the gateway CLI after installing the adapted extra and pinned model. [Full setup, interpretation limits and tutoring example](CONTEXT.md). Model outputs never grant source permissions.
