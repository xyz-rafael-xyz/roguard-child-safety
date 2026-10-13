# Interaction laboratory

RoGuard 5 connects five functions: replay a fictional interaction, track information and permissions through transformations, propose a minimum disclosure, test decision consistency, and measure permitted tasks completed alongside unauthorized proposals. The existing v4 text classifiers are unchanged. A programmed reference target is not a trained model, and the new laboratory results are not new child-language accuracy results.

## Start the browser workspace

```sh
pip install --upgrade roguard-child-safety
roguard-studio --lab --open
```

Choose Romanian or Ukrainian, a fictional situation and a reference behavior. Run the simulation. Click a request in the timeline to inspect its minimum-disclosure proposal. Enable reduction to find a smaller failing scenario. Compare behaviors to see why refusing everything does not preserve useful assistance. All browser actions are simulated; there is no live tool execution route or remote model URL field.

The four programmed reference behaviors are:

- **Guarded:** follows the minimum permitted disclosure proposal and performs the checkable arithmetic task.
- **Permissive:** proposes every candidate field, even if a source is restricted.
- **Overblocking:** refuses every request.
- **Stale:** reuses its first proposal, ignoring subsequent changes.

Disabling enforcement changes only the simulation. It cannot disable the separate SDK gateway. The classifier-variation button uses Studio's active `fast`, `semantic` or `hybrid` backend; the result names the backend. To use the optional semantic model, install `roguard-child-safety[semantic]`, run `roguard-model fetch`, and start `roguard-studio --lab --open --backend semantic`. See the [conversation setup guide](CONVERSATION_WORKSPACE.md) for its download, Python requirements and limits.

The optional JSON editor accepts the [scenario schema](../schema/interaction-scenario.schema.json). It does not execute code. After an input changes, previous results and exports are invalidated. Clear removes current browser inputs and results. The application does not use persistent browser storage. Ordinary reports omit message and artifact values. **Reduced reproducer exports contain the supplied scenario text.** Use fictional data; browser extensions, caller logs and downloaded files are outside this application's control.

## CLI and CI

The demo and built-in suite work immediately after installation:

```sh
roguard-lab demo --language uk --family revocation --target stale
roguard-lab compare --language ro
roguard-lab suite --seed 17 --repetitions 2 --fail-on-failure
```

The following examples use files in the GitHub checkout:

```sh
roguard-lab run examples/lab_revocation_ro.json --recorded examples/lab_recorded_replies.json
roguard-lab reduce examples/lab_revocation_uk.json --target stale --max-probes 96 > reproducer-report.json
roguard-lab consistency examples/lab_surface_conversations.json --backend fast
```

`--fail-on-failure` is available for `demo`, `run`, `suite` and `consistency`. It prints the JSON report and exits with status 1 on a reported failure; consistency also fails this gate for unresolved pairs. Without it, the commands report findings without failing the process. Reference comparison intentionally includes failing baselines. A gate passing a declared test does not approve deployment.

`run_scenario(scenario, target_factory, enforce=True)` connects another chatbot through a **synchronous Python callable**. The factory must create a fresh target instance for each replay. It receives an observation with fictional messages, artifact values, current request, clock and a disclosure proposal. The `expected` test label and branch targets are withheld. The target returns:

```json
{"decision": "ALLOW", "selected": ["summary"], "answer": 12}
```

`decision` is `ALLOW`, `BLOCK` or `REVIEW`. Only `ALLOW` may select nonempty candidate IDs. `answer` is optional and must be an integer for the supported arithmetic task. Extra fields—including proposed permission changes—are rejected. Exceptions and invalid replies become unresolved findings without echoing exception text. The target is never allowed to mutate the ledger through its observation. An adapter may explicitly call a model service under the host application's own authorization; RoGuard does not choose or contact one automatically. The simulator is not a sandbox for that Python callable.

`recorded_factory(replies)` replays a list of previously collected structured replies without a live model. Replies are consumed sequentially. When reducing a scenario, a real adapter must answer the changed observations; a recorded adapter only tests that fixed response sequence and cannot establish how the original model would respond to the reduced input.

## Information that retains its restrictions

A `FlowLedger` holds values in memory for one session. Source IDs are immutable and opaque. Each source starts without permission. The trusted host declares a grant for one **source, purpose, recipient and operation**. Operations are `use`, `memory`, `export` and `tool`. Grants can expire, pause, resume or be revoked. The monotonic logical clock uses host-supplied integer ticks; applications must map real time to ticks correctly and keep the clock current. This is not a wall-clock scheduler.

A derived summary, proposed memory or tool result inherits the union of its parents' source IDs. Every source must permit a proposed use, and every ancestor must have complete declared provenance. A later revocation therefore affects already-created descendants. A grant on one purpose or recipient does not authorize another, and derived artifacts cannot receive a new grant that bypasses their sources. Registering a proposed memory artifact is not permission to persist it; persistence must cross a gateway binding with operation `memory`.

```python
from roguard import FlowLedger, minimize_disclosure

with FlowLedger() as ledger:
    ledger.add_source("context", "Fictional learning preference.")
    ledger.add_source("private_note", "Fictional note for a different task.")
    ledger.derive("short_summary", "Fictional short summary.", ["context"])
    ledger.permission("context", "tutoring", "student", "use", "grant")
    plan = minimize_disclosure(
        ledger,
        [
            {"artifact_id": "short_summary", "covers": ["task_context"], "cost": 1},
            {"artifact_id": "context", "covers": ["task_context"], "cost": 4},
            {"artifact_id": "private_note", "covers": ["task_context"], "cost": 8},
        ],
        ["task_context"], purpose="tutoring", recipient="student", operation="use",
    )
    assert plan["selected"] == ["short_summary"]
```

The host must register **all** source dependencies and keep data inside instrumented boundaries. The ledger cannot discover an omitted source, inspect arbitrary generated text for hidden provenance, prevent a trusted callback from leaking data, or retract an earlier disclosure. Unknown external tool output is marked incomplete by default. `close()` drops retained values and permissions; this is not a guarantee of cryptographic memory erasure.

## Exact minimum disclosure

`minimize_disclosure` accepts up to 18 candidate artifacts and 16 required goal IDs. It excludes candidates lacking current permission, then exactly minimizes total declared cost, field count and finally lexical ID order. A zero cost does not bypass permissions. The result explains retained requirements, excluded candidates and any missing coverage. If no permitted subset covers every requirement, it returns no selection.

Optimality applies only to the supplied candidate pool and cost model. A trusted caller declares which requirements each candidate covers: this is not a semantic verifier or a guarantee that a summary is useful. The plan is a proposal, **not an execution capability**. The gateway must recheck the current ledger when delivering its selected fields.

## Host-bound execution gateway

Only the Python SDK can invoke a host-configured callback. The callback's purpose, recipient and operation are fixed by `ToolBinding`; model JSON cannot redirect the destination. Dispatch is a dry run unless `execute=True` is explicitly passed. `export` and `tool` bindings require host approval; local `use` or `memory` bindings can be configured otherwise by the trusted host.

```python
from roguard import FlowLedger, ToolBinding, ToolGateway

received = []  # Local capture only; this example sends nothing externally.
with FlowLedger() as ledger:
    ledger.add_source("summary", "Fictional summary for a reviewer.")
    ledger.permission("summary", "review", "reviewer", "export", "grant")
    gateway = ToolGateway(ledger, {
        "handoff": ToolBinding("review", "reviewer", "export", received.append),
    })
    approval = gateway.approve("handoff", ["summary"])  # Trusted host attestation.
    result = gateway.dispatch("handoff", ["summary"], approval=approval, execute=True)
    assert result["callback_invoked"]
    assert len(received) == 1
```

Approval is an opaque in-process object bound to the gateway, tool, exact selected artifact IDs and policy revision. Any ledger change invalidates it; executing consumes it, including when a callback fails. A model-supplied boolean is not an approval token. `approve()` records a trusted host attestation; it does not authenticate a human or verify legal authority.

Checks and callback invocation are serialized under the ledger lock. Callbacks must be trusted and synchronous, must not mutate the ledger, and must not retain untracked copies. A remote service or a callback that queues work needs its own final atomic authorization boundary: the local lock cannot revoke a remotely queued action. On callback failure, the report says `EXECUTION_UNCERTAIN`, since a partial side effect may have occurred; the SDK never retries automatically. `callback_invoked` is the observable fact, while `external_action_taken` is unknown for executed callbacks.

With `output_id`, returned text can be registered as a derived `tool_result`. By default its provenance is incomplete, because an external service may introduce new information. Only a trusted, verified transformation of the supplied inputs should set `ToolBinding(..., output_provenance_complete=True)`. Invalid output after execution is reported as uncertain, without pretending the callback was rolled back.

## Consistency and usefulness

`evaluate_consistency(pairs, predictor)` accepts arbitrary caller-supplied paired tests. An invariant pair asks whether a decision stays stable; a transition specifies two different required endpoint decisions. Expected labels are optional for invariance but mandatory for transitions. Equal wrong decisions fail when expected labels exist. If either side is unresolved, the pair is unresolved rather than correct. The predictor receives a deep copy of each input. Paired equivalence is a declared hypothesis, not independently verified meaning.

`conversation_consistency(payloads, scorer)` builds five surface variants per Romanian/Ukrainian input: capitalization, spacing, Unicode composition, punctuation and missing Romanian diacritics or Ukrainian mixed Latin `i`. It reports the active backend. It does not translate text, tune models, or make invariance a substitute for accuracy. The browser's six policy pairs are separate controlled tests; custom scenarios and recorded replies do not automatically inherit them.

`summarize_runs` reports explicit numerators and denominators for unauthorized proposals, unauthorized simulated deliveries, stopped proposals, permitted-task preservation, unnecessary refusal, unresolved requests and checkable arithmetic success. A task is completed only with allowed delivery, required field coverage and a correct arithmetic answer when applicable. A missing denominator produces `null`, not a fabricated perfect rate. `tradeoff_frontier` compares measured violation/utility rates under a caller-declared ceiling; it does not choose a deployment policy.

## Limits and evidence

The [release report](V5_RELEASE.md) and [registered protocol](V5_LAB_PROTOCOL.md) give reference comparisons and bounded transition checks. The [v4 report](V4_RELEASE.md) remains the source for unchanged classifier accuracy. Simulated message text, policy facts, goal coverage and costs are project-authored. There is no new independent language review or evidence of authentic-child performance. The distinct contribution is the connected workflow and inspectable failure/permission/utility records; the underlying methods have prior art.
