"""Experimental durable, host-owned conversation review sessions.

The host owns authentication, filesystem access, retention and reviewer authority.
This local SQLite store performs no network calls or external actions. Model
outputs are untrusted observations: an abstention or positive creates an open
review reference, and subsequent negative predictions cannot close it.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3

from .assurance import digest
from .trajectory_data import input_identity, validate_conversation


class SessionConflict(ValueError):
    """A revision, request identity, context, or model binding changed."""


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def _identifier(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 200 or not value.strip():
        raise ValueError('Expected a nonempty identifier of at most 200 characters')
    return value


def _revision(value):
    if type(value) is not int or value < 0:
        raise ValueError('Expected a nonnegative integer revision')


class ReviewSessionStore:
    """One atomic revision per append or host resolution, with idempotent retries.

    Use a file path to survive restarts. Each operation opens its own connection;
    separate processes can share that file. Inference runs outside the write
    transaction, then commits only if the original revision still matches.
    A model failure aborts the entire append. Callers must surface such failures.

    ``model_identity`` must bind the complete predictor configuration, including
    thresholds, memory and calibration, not merely a weight file. A predictor
    receives only a language and a prefix of validated turns, and returns a dict
    with the matching input_digest and predicted equal to 0, 1, or None.
    """

    def __init__(self, path):
        self.path = str(path)
        if self.path == ':memory:':
            raise ValueError('Use a persistent SQLite file')
        with self._connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS trajectory_sessions (
                    id TEXT PRIMARY KEY, model TEXT NOT NULL, revision INTEGER NOT NULL,
                    state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trajectory_requests (
                    session TEXT NOT NULL, request TEXT NOT NULL, signature TEXT NOT NULL,
                    response TEXT NOT NULL, PRIMARY KEY (session, request),
                    FOREIGN KEY (session) REFERENCES trajectory_sessions(id)
                );
            ''')

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
        finally:
            if db.in_transaction:
                db.rollback()
            db.close()

    def create(self, session_id, language, *, model_identity):
        _identifier(session_id)
        _identifier(model_identity)
        if language not in ('ro', 'uk'):
            raise ValueError('Expected ro or uk')
        state = {'language': language, 'turns': [], 'open': {}, 'resolved': [],
                 'last_prediction': None}
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT model, state FROM trajectory_sessions WHERE id=?',
                             (session_id,)).fetchone()
            if old:
                if old[0] != model_identity or json.loads(old[1])['language'] != language:
                    raise SessionConflict('Session already has another binding')
            else:
                db.execute('INSERT INTO trajectory_sessions VALUES (?, ?, 0, ?)',
                           (session_id, model_identity, _json(state)))
            db.commit()
        return self.read(session_id)

    @staticmethod
    def _load(db, session_id):
        row = db.execute('SELECT model, revision, state FROM trajectory_sessions WHERE id=?',
                         (session_id,)).fetchone()
        if row is None:
            raise KeyError('Unknown session')
        return row[0], row[1], json.loads(row[2])

    @staticmethod
    def _snapshot(session_id, model, revision, state):
        context = ({'language': state['language'], 'turns': state['turns']}
                   if state['turns'] else None)
        return {'format': 'roguard-durable-session-v10-research1',
                'session_id': session_id, 'model_identity': model, 'revision': revision,
                'language': state['language'], 'turn_count': len(state['turns']),
                'context_digest': input_identity(context) if context else None,
                'unresolved_review_references': sorted(state['open'].values(),
                    key=lambda entry: (entry['focus'], entry['reference'])),
                'session_review_required': bool(state['open']),
                'last_prediction': state['last_prediction'],
                'qualified': False, 'establishes_safety': False,
                'permits_external_action': False}

    def read(self, session_id):
        _identifier(session_id)
        with self._connection() as db:
            model, revision, state = self._load(db, session_id)
        return self._snapshot(session_id, model, revision, state)

    @staticmethod
    def _retry(db, session_id, request_id, signature):
        row = db.execute('SELECT signature, response FROM trajectory_requests '
                         'WHERE session=? AND request=?', (session_id, request_id)).fetchone()
        if row is None:
            return None
        if row[0] != signature:
            raise SessionConflict('Request identifier reused with different content')
        return json.loads(row[1])

    def _commit(self, session_id, model, revision, state, request_id, signature, response):
        serialized = _json(state)
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            retry = self._retry(db, session_id, request_id, signature)
            if retry is not None:
                return retry
            updated = db.execute('UPDATE trajectory_sessions SET state=?, revision=revision+1 '
                                 'WHERE id=? AND model=? AND revision=?',
                                 (serialized, session_id, model, revision)).rowcount
            if updated != 1:
                raise SessionConflict('Session advanced during operation; read and retry')
            db.execute('INSERT INTO trajectory_requests VALUES (?, ?, ?, ?)',
                       (session_id, request_id, signature, _json(response)))
            db.commit()
        return response

    def append(self, session_id, turns, *, expected_revision, request_id,
               model_identity, predictor):
        """Evaluate EVERY appended minor prefix; atomically persist all or none.

        A successful duplicate returns the original response without inference.
        Responses include committed_revision; after a retry read() obtains the
        latest session, which may have advanced since that original response.
        """
        _identifier(session_id)
        _identifier(request_id)
        _identifier(model_identity)
        _revision(expected_revision)
        # Own the caller's data before invoking arbitrary prediction code.
        turns = json.loads(_json(turns))
        signature = digest({'operation': 'append', 'turns': turns,
                            'revision': expected_revision, 'model': model_identity})
        with self._connection() as db:
            retry = self._retry(db, session_id, request_id, signature)
            if retry is not None:
                return retry
            model, revision, state = self._load(db, session_id)
        if revision != expected_revision or model != model_identity:
            raise SessionConflict('Stale revision or changed predictor configuration')
        validate_conversation({'language': state['language'], 'turns': turns})
        old_count = len(state['turns'])
        state['turns'].extend(turns)
        validate_conversation({'language': state['language'], 'turns': state['turns']})
        predictions = []
        for focus in range(old_count, len(state['turns'])):
            if state['turns'][focus]['role'] != 'minor':
                continue
            payload = {'language': state['language'], 'turns': state['turns'][:focus+1]}
            identity = input_identity(payload)
            result = predictor(json.loads(_json(payload)))
            if not isinstance(result, dict) or result.get('input_digest') != identity:
                raise ValueError('Prediction is not bound to the requested prefix')
            value = result.get('predicted', 'missing')
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError('Prediction must be 0, 1, or an explicit abstention')
            # Persist bounded observations; model-generated instructions, action
            # permissions or invented session fields are never authoritative.
            observation = {'focus': focus, 'input_digest': identity, 'predicted': value}
            predictions.append(observation)
            state['last_prediction'] = observation
            if value != 0:
                state['open'][identity] = {'reference': identity, 'focus': focus,
                    'reason': 'MODEL_ABSTAINED' if value is None else 'MODEL_REVIEW',
                    'opened_revision': revision+1}
        response = {'operation': 'append', 'committed_revision': revision+1,
                    'predictions': predictions,
                    'session': self._snapshot(session_id, model, revision+1, state)}
        return self._commit(session_id, model, revision, state, request_id, signature, response)

    def resolve(self, session_id, reference, *, expected_revision,
                expected_context_digest, request_id, reviewer_id, reason):
        """Trusted host/reviewer operation; never expose this to the model.

        The host must authenticate and authorize reviewer_id. Supplying a name
        here is an audit record, not proof of authorization.
        """
        for value in (session_id, reference, request_id, reviewer_id, expected_context_digest):
            _identifier(value)
        _revision(expected_revision)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError('A reviewer resolution needs a reason of at most 2000 characters')
        signature = digest({'operation': 'resolve', 'reference': reference,
            'revision': expected_revision, 'context': expected_context_digest,
            'reviewer': reviewer_id, 'reason': reason})
        with self._connection() as db:
            retry = self._retry(db, session_id, request_id, signature)
            if retry is not None:
                return retry
            model, revision, state = self._load(db, session_id)
        if revision != expected_revision or not state['turns'] or input_identity(
                {'language': state['language'], 'turns': state['turns']}) != expected_context_digest:
            raise SessionConflict('Reviewer resolution addresses stale context')
        if reference not in state['open']:
            raise SessionConflict('Unknown or already resolved reference')
        state['resolved'].append({**state['open'].pop(reference), 'reviewer_id': reviewer_id,
            'reason': reason, 'resolved_revision': revision+1,
            'resolution_context_digest': expected_context_digest})
        response = {'operation': 'resolve', 'committed_revision': revision+1,
                    'session': self._snapshot(session_id, model, revision+1, state)}
        return self._commit(session_id, model, revision, state, request_id, signature, response)

    def audit(self, session_id):
        """Return original context and host resolutions to an authorized caller."""
        _identifier(session_id)
        with self._connection() as db:
            model, revision, state = self._load(db, session_id)
        return {'model_identity': model, 'revision': revision, **state}
