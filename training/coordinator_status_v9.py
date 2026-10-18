"""Record terminal setup failures so dependent research jobs do not wait forever."""
from datetime import datetime, timezone
import json
from pathlib import Path


def supervise(function, study):
    try:
        return function()
    except Exception as error:
        study=Path(study);study.mkdir(parents=True,exist_ok=True)
        path=study/'run-status.json'
        try:
            status=json.loads(path.read_text()) if path.exists() else {}
        except (OSError,json.JSONDecodeError):
            status={}
        if not isinstance(status,dict):status={}
        if not isinstance(status.get('jobs'),list):status['jobs']=[]
        now=datetime.now(timezone.utc).isoformat()
        status['jobs'].append(dict(name='coordinator',exit_code=1,error_type=type(error).__name__,finished_utc=now))
        status.update(complete=True,finished_utc=now,stable_release_ready=False,reserved_probes_used=False)
        temporary=study/'run-status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(path)
        raise
