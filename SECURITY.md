# Security

For a vulnerability in code or the review gate, use GitHub's [private vulnerability report](https://github.com/xyz-rafael-xyz/roguard-child-safety/security/advisories/new) before public disclosure. Do not open a public issue with exploit details. Use only abstract, synthetic reproduction inputs; never include child data, example disclosures, private reviewer material, credentials, or personal data. The packaged browser and CLI have no live escalation or reporting integration. The v5 Python SDK can invoke explicitly registered trusted callbacks; see the boundary below.

## Interaction SDK boundary

The host is trusted to register complete source provenance, authoritative grants, current logical time, goal coverage, costs and callback destinations. Arbitrary Python callers can bypass private attributes; the ledger is not a sandbox or a multi-tenant access-control service. Callback code must be synchronous, must not mutate the ledger, and must not retain untracked copies. External services and queues require their own final authorization check.

External-operation bindings require opaque single-use host approval at the current ledger revision. Models must never receive the host permission/approval methods as tools. Dry runs do not call callbacks. Failed callbacks are reported as uncertain and never automatically retried. Outputs with unverified additional provenance are incomplete by default. Prior disclosure cannot be undone by revocation.

The browser's scenario JSON cannot load code, set a target URL or dispatch a registered callback. Existing loopback host, origin, session-token, size-limit and content-security protections cover the laboratory routes. Ordinary reports omit message values; reduced reproducer exports contain input text. Use synthetic examples when reporting a vulnerability.

## V6 durable service and adapters

The optional service authenticates tenant-scoped reader/operator/admin bearer credentials. Protect and rotate those credentials and the separate encryption key; model tools must never hold admin credentials. State values and policy journals are encrypted, while SQLite metadata and fixed monitoring fields are not. The bounded service requires deployment-owned TLS and access controls when exposed beyond loopback. Callbacks are trusted, synchronous, explicitly registered host code. No route loads Python or chooses destinations. Change `binding_version` whenever a callback or its destination/configuration changes.

Execution intent is persisted before callback invocation. A crash or thrown callback leaves uncertainty and never triggers an automatic retry. Cancellation cannot undo an invoked callback. Remote systems must implement their own final authorization and idempotency. SQLite serialization is local to the shared file, not a distributed transaction. See [the service limits and retention behavior](docs/RELIABILITY.md).

The adapted encoder uses the pinned base and a hash-verified JSON tensor artifact. Model signals, calibration sets, support distances and long-context windows remain advisory and cannot create authority. Network model adapters require explicit endpoints and credentials and transmit supplied observation content to those endpoints; the default workspace has no such network adapter.
