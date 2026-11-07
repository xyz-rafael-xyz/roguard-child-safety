# Durable review sessions (experimental)

`ReviewSessionStore` adds a local, persistent conversation workflow to the v10
research branch. It does not qualify a model or establish that a conversation is
safe. The published package remains v9 until the release criteria are met.

The host app supplies its configured predictor. Each append checks **every new
minor turn**, using only the conversation prefix ending at that turn. A later
ordinary message cannot erase an earlier positive prediction or abstention.
The full validated transcript and original prefix references survive restarts.
Only an explicit host reviewer resolution can close a reference.

```python
from roguard.trajectory_store import ReviewSessionStore, SessionConflict

# Bind the complete pipeline, including the chosen calibration and all settings.
from roguard.assurance import digest
binding = digest({
    "pipeline": model.pipeline_digest(),
    "calibration": calibration,
    "plan": True,
})

def predict_prefix(payload):
    return model.predict(payload, calibration=calibration, plan=True)

store = ReviewSessionStore("review-sessions.sqlite")
session = store.create("conversation-123", "ro", model_identity=binding)
result = store.append(
    "conversation-123",
    [{"role": "minor", "text": "Am nevoie să vorbesc cu cineva."}],
    expected_revision=session["revision"],
    request_id="message-001",  # persist and reuse this ID when retrying
    model_identity=binding,
    predictor=predict_prefix,
)
session = result["session"]
```

On a network retry, send exactly the same request ID, content and expected
revision. The store returns the original response without repeating inference.
Its `committed_revision` identifies that original operation; call `read()` to get
the latest state if another operation may have advanced the session. Reusing a
request ID for changed content raises `SessionConflict`.

Inference runs outside SQLite's write transaction. The final commit checks the
original revision again. Competing writers cannot silently overwrite each other.
After a conflict, read the current session and explicitly rebase the intended
new operation with a new request ID. If the predictor raises or returns an invalid
or incorrectly bound prediction, the entire batch is rejected without advancing
the session. The host must display the failed operation and retry it; a failed
append is not a negative prediction.

A reviewer resolution must address the latest revision and context:

```python
session = store.read("conversation-123")
reference = session["unresolved_review_references"][0]["reference"]
result = store.resolve(
    "conversation-123", reference,
    expected_revision=session["revision"],
    expected_context_digest=session["context_digest"],
    request_id="review-001",
    reviewer_id=authenticated_reviewer.id,
    reason="Reviewed the original context and completed the host review process.",
)
```

The host must authenticate and authorize the reviewer. A supplied reviewer name
is an audit record, not proof of permission. Do not expose `resolve()` as a model
tool. Predictions cannot grant external-action permission or change session state.
`audit()` returns the original transcript and resolution records to an authorized
caller. The store does not send messages or contact anyone.

The database contains conversation text in plaintext. The host owns filesystem
access, retention, deletion, backups and authorization for every store method.
Use a local SQLite-compatible filesystem and a separate database for isolated
tenants when appropriate. Sessions accept Romanian or Ukrainian, at most 512
turns and the existing conversation size limits. There is no silent eviction of
old context or unresolved references. SQLite transactions protect ordinary
concurrent operations; this API does not claim to detect malicious file edits.

Software tests cover process restart, concurrent appends, duplicate requests,
inference failure, malformed predictions, changed model bindings, stale reviewer
resolutions and resolution/append races. These verify workflow behavior, not
classifier accuracy or benefit to reviewers. The in-memory `TrajectorySession`
is still available, but the durable API is the one that checks every minor prefix
in an appended batch.
