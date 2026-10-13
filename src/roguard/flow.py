"""In-memory provenance and host-declared information-flow constraints.

The host is trusted to register complete provenance and authoritative policy.
This module is not a sandbox for callbacks or arbitrary Python callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable
import re

OPERATIONS = frozenset({'use', 'memory', 'export', 'tool'})


def identifier(value, name='identifier'):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError(f'Invalid {name}; use an opaque ASCII identifier')
    return value


def identifiers(values, name='identifiers', maximum=32, allow_empty=False):
    if not isinstance(values, (list, tuple)) or len(values) > maximum or (not values and not allow_empty):
        raise ValueError(f'Invalid {name}')
    result = tuple(identifier(v, name) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(f'Duplicate {name}')
    return result


def integer(value, name, maximum=10**12):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f'Invalid {name}')
    return value


def scope(purpose, recipient, operation):
    identifier(purpose, 'purpose'); identifier(recipient, 'recipient')
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError('Unknown operation')
    return purpose, recipient, operation


@dataclass(frozen=True)
class _Artifact:
    value: str
    parents: tuple
    roots: frozenset
    complete: bool
    kind: str


class FlowLedger:
    """Session-scoped data; close() drops retained values and permissions."""

    def __init__(self):
        self._artifacts = {}
        self._permissions = {}
        self._clock = 0
        self._revision = 0
        self._closed = False
        self._lock = RLock()

    def _open(self):
        if self._closed:
            raise ValueError('Ledger is closed')

    @property
    def clock(self):
        return self._clock

    @property
    def revision(self):
        return self._revision

    def _new(self, artifact_id, value):
        self._open(); identifier(artifact_id, 'artifact')
        if artifact_id in self._artifacts or len(self._artifacts) >= 512:
            raise ValueError('Duplicate artifact or session artifact limit reached')
        if not isinstance(value, str) or not value.strip() or len(value) > 6000:
            raise ValueError('Artifact values must contain 1 to 6000 characters')

    def add_source(self, artifact_id, value, *, complete=True):
        with self._lock:
            self._new(artifact_id, value)
            if type(complete) is not bool:
                raise ValueError('Provenance completeness must be boolean')
            self._artifacts[artifact_id] = _Artifact(value, (), frozenset({artifact_id}), complete, 'source')
            self._revision += 1

    def derive(self, artifact_id, value, parents, *, kind='summary', complete=True):
        """Register a proposed transformation; this does not authorize its use."""
        with self._lock:
            self._new(artifact_id, value)
            parents = identifiers(parents, 'parents')
            if any(p not in self._artifacts for p in parents):
                raise ValueError('Unknown parent artifact')
            if kind not in {'summary', 'memory', 'tool_result', 'transformation'} or type(complete) is not bool:
                raise ValueError('Invalid transformation declaration')
            roots = frozenset().union(*(self._artifacts[p].roots for p in parents))
            complete = complete and all(self._artifacts[p].complete for p in parents)
            self._artifacts[artifact_id] = _Artifact(value, parents, roots, complete, kind)
            self._revision += 1

    def permission(self, root, purpose, recipient, operation, action, *, expires_at=None):
        """Trusted host operation. Never expose this method as a model tool."""
        with self._lock:
            self._open(); key = (identifier(root, 'source'),) + scope(purpose, recipient, operation)
            if root not in self._artifacts or self._artifacts[root].kind != 'source':
                raise ValueError('Permissions must address a registered source')
            if action not in {'grant', 'revoke', 'pause', 'resume'}:
                raise ValueError('Unknown permission action')
            if expires_at is not None:
                integer(expires_at, 'expiry')
                if action != 'grant' or expires_at <= self.clock:
                    raise ValueError('Only a future grant expiry can be set')
            if key not in self._permissions and len(self._permissions) >= 4096:
                raise ValueError('Session permission limit reached')
            state, expiry = self._permissions.get(key, ('absent', None))
            if action == 'grant':
                state, expiry = 'active', expires_at
            elif action == 'revoke':
                state, expiry = 'absent', None
            elif action == 'pause' and state == 'active':
                state = 'paused'
            elif action == 'resume' and state == 'paused':
                state = 'active'
            self._permissions[key] = state, expiry
            self._revision += 1

    def advance(self, at):
        with self._lock:
            self._open(); integer(at, 'clock')
            if at < self.clock:
                raise ValueError('Clock cannot move backwards')
            self._clock = at
            self._revision += 1

    def describe(self, artifact_id):
        with self._lock:
            self._open(); identifier(artifact_id)
            artifact = self._artifacts[artifact_id]
            return {'artifact_id': artifact_id, 'parents': list(artifact.parents),
                    'sources': sorted(artifact.roots), 'provenance_complete': artifact.complete,
                    'kind': artifact.kind}

    def check(self, artifact_ids, purpose, recipient, operation):
        with self._lock:
            self._open(); artifact_ids = identifiers(artifact_ids, 'artifacts')
            context = scope(purpose, recipient, operation)
            reasons = set(); roots = set()
            for name in artifact_ids:
                artifact = self._artifacts.get(name)
                if artifact is None:
                    reasons.add('UNKNOWN_ARTIFACT'); continue
                roots.update(artifact.roots)
                if not artifact.complete:
                    reasons.add('INCOMPLETE_PROVENANCE')
            for root in roots:
                state, expiry = self._permissions.get((root,) + context, ('absent', None))
                if state != 'active':
                    reasons.add('SOURCE_PERMISSION_INACTIVE')
                if expiry is not None and self.clock >= expiry:
                    reasons.add('SOURCE_PERMISSION_EXPIRED')
            return {'decision': 'BLOCK' if reasons else 'ALLOW', 'reason_codes': sorted(reasons),
                    'sources': sorted(roots), 'artifact_ids': list(artifact_ids),
                    'clock': self.clock, 'policy_revision': self.revision,
                    'authority_verified': False, 'text_in_report': False}

    def close(self):
        with self._lock:
            self._artifacts.clear(); self._permissions.clear(); self._closed = True

    def __enter__(self):
        self._open(); return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class ToolBinding:
    purpose: str
    recipient: str
    operation: str
    callback: Callable
    requires_approval: bool = True
    output_provenance_complete: bool = False

    def __post_init__(self):
        scope(self.purpose, self.recipient, self.operation)
        if not callable(self.callback) or type(self.requires_approval) is not bool or type(self.output_provenance_complete) is not bool:
            raise ValueError('Invalid trusted tool binding')
        if self.operation in {'export', 'tool'} and not self.requires_approval:
            raise ValueError('External operation bindings require host approval')


class ToolGateway:
    """Opt-in host callbacks. Destinations are bound by the host, not the model.

    The ledger lock spans checks and invocation. Callbacks must be trusted,
    synchronous and must not modify the ledger or retain untracked copies.
    External systems need their own atomic authorization boundary.
    """

    def __init__(self, ledger, bindings):
        if not isinstance(ledger, FlowLedger) or not isinstance(bindings, dict):
            raise ValueError('Expected ledger and trusted bindings')
        if any(identifier(name) != name or not isinstance(binding, ToolBinding)
               for name, binding in bindings.items()):
            raise ValueError('Invalid tool registry')
        self.ledger = ledger
        self._bindings = dict(bindings)
        self._approvals = {}

    def approve(self, tool, artifact_ids):
        """Trusted host attestation, bound to tool, artifacts and policy revision.

        Tokens are opaque Python objects, never model-readable JSON authority.
        This does not verify who the human is or establish legal permission.
        """
        identifier(tool, 'tool')
        if tool not in self._bindings:
            raise ValueError('Unknown host-bound tool')
        ids = identifiers(artifact_ids, 'artifacts')
        binding = self._bindings[tool]
        with self.ledger._lock:
            report = self.ledger.check(ids, binding.purpose, binding.recipient, binding.operation)
            if report['decision'] != 'ALLOW':
                raise ValueError('Cannot approve an ineligible proposal')
            # Expired approvals cannot accumulate across policy revisions.
            self._approvals = {token: record for token, record in self._approvals.items()
                               if record[2] == self.ledger.revision}
            if len(self._approvals) >= 1024:
                raise ValueError('Outstanding approval limit reached')
            token = object()
            self._approvals[token] = (tool, tuple(sorted(ids)), self.ledger.revision)
            return token

    def dispatch(self, tool, artifact_ids, *, approval=None, execute=False, output_id=None):
        if type(execute) is not bool:
            raise ValueError('Execution must be an explicit boolean')
        identifier(tool, 'tool')
        if tool not in self._bindings:
            raise ValueError('Unknown host-bound tool')
        binding = self._bindings[tool]
        with self.ledger._lock:
            report = self.ledger.check(artifact_ids, binding.purpose, binding.recipient, binding.operation)
            report.update(tool=tool, external_action_taken=False, callback_invoked=False)
            if report['decision'] == 'BLOCK':
                return report
            approved = (type(approval) is object and self._approvals.get(approval) ==
                        (tool, tuple(sorted(artifact_ids)), self.ledger.revision))
            if binding.requires_approval and not approved:
                report.update(decision='REVIEW', reason_codes=['CURRENT_HOST_APPROVAL_REQUIRED'])
                return report
            if not execute:
                report.update(decision='READY', reason_codes=['DRY_RUN_ONLY'])
                return report
            if output_id is not None:
                self.ledger._new(output_id, 'reserved')
            if approved:
                self._approvals.pop(approval)
            values = {name: self.ledger._artifacts[name].value for name in artifact_ids}
            # Failure cannot prove absence of a side effect. Never retry implicitly.
            try:
                result = binding.callback(values)
            except Exception:
                report.update(decision='EXECUTION_UNCERTAIN', reason_codes=['CALLBACK_FAILED_NO_AUTOMATIC_RETRY'],
                              callback_invoked=True, external_action_taken=None)
                return report
            report.update(decision='EXECUTED', reason_codes=[], callback_invoked=True,
                          external_action_taken=None)
            if output_id is not None:
                try:
                    self.ledger.derive(output_id, result, list(artifact_ids), kind='tool_result', complete=binding.output_provenance_complete)
                    report['output_artifact'] = output_id
                except (ValueError, TypeError):
                    report.update(decision='EXECUTION_UNCERTAIN', reason_codes=['UNTRACKED_CALLBACK_RESULT'])
            return report
