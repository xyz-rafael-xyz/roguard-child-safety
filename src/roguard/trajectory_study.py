"""Hash-bound, single-opening final-test workflow, separate from development.

Files and declared authors must be controlled by an independent study custodian.
Local hashes do not prove blindness, identity, or that somebody never read a file.
"""
from __future__ import annotations

from datetime import datetime,timezone
import json
import os
from pathlib import Path

from .assurance import digest
from .trajectory_data import file_sha,read_jsonl,audit_splits


def freeze(*,artifacts,test_file,output,exposure_log=None):
    output=Path(output)
    if output.exists():raise ValueError('Freeze already exists')
    if not artifacts or len(artifacts)!=len(set(artifacts)):raise ValueError('Provide unique model/pipeline artifacts')
    # Paths stay in this local custodian manifest, never a public report.
    log=Path(exposure_log) if exposure_log is not None else output.with_name(output.name+'.exposure.json')
    if log.exists():raise ValueError('Exposure log already exists')
    if log.resolve() in {output.resolve(),Path(test_file).resolve(),*(Path(p).resolve() for p in artifacts)}:
        raise ValueError('Exposure log must have its own path')
    document={'format':'roguard-v10-final-freeze-2','created_utc':datetime.now(timezone.utc).isoformat(),
              'artifacts':{str(Path(p).resolve()):file_sha(p) for p in artifacts},
              'test_file':str(Path(test_file).resolve()),'test_sha256':file_sha(test_file),
              'exposure_log':str(log.resolve()),
              'independence_verified':False,'test_content_parsed':False}
    document['digest']=digest(document)
    with output.open('x',encoding='utf-8') as f:json.dump(document,f,indent=2)
    return {'freeze_digest':document['digest'],'test_sha256':document['test_sha256'],'test_content_parsed':False}


def open_once(freeze_file,*,exposure_log=None,development_cases):
    document=json.loads(Path(freeze_file).read_text())
    raw={k:v for k,v in document.items() if k!='digest'}
    if document.get('digest')!=digest(raw) or document.get('format')!='roguard-v10-final-freeze-2':
        raise ValueError('Invalid final-test freeze')
    log=Path(document['exposure_log'])
    if exposure_log is not None and Path(exposure_log).resolve()!=log:
        raise ValueError('Exposure log differs from the frozen path')
    for path,expected in document['artifacts'].items():
        if file_sha(path)!=expected:raise ValueError('Model or pipeline changed after freeze')
    if file_sha(document['test_file'])!=document['test_sha256']:raise ValueError('Final-test file changed')
    exposure={'freeze_digest':document['digest'],'opened_utc':datetime.now(timezone.utc).isoformat(),
              'test_sha256':document['test_sha256'],'consumed_even_if_run_fails':True}
    try:
        with log.open('x',encoding='utf-8') as f:
            json.dump(exposure,f,indent=2);f.flush();os.fsync(f.fileno())
    except FileExistsError as error:raise ValueError('Final test already exposed; it cannot become fresh again') from error
    rows=read_jsonl(document['test_file'])
    if any(r['split']!='test' for r in rows):raise ValueError('Final test must be declared test data')
    audit_splits(list(development_cases)+rows,independent_test=True)
    return rows,exposure
