import json

import pytest

from scripts.run_large_data_sequence_v10 import check_sources,completed_dependency,sha


def test_queued_commands_require_unchanged_source_and_completed_dependency(tmp_path):
    source=tmp_path/'fit.py';source.write_text('pass\n')
    report=tmp_path/'report.json';plan={'cwd':str(tmp_path),'source_hashes':{'fit.py':sha(source)},'dependency_report':str(report)}
    check_sources(plan)
    with pytest.raises(ValueError,match='Dependency'):completed_dependency(plan)
    report.write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError,match='Dependency'):completed_dependency(plan)
    report.write_text(json.dumps({'complete':True,'decision':'REJECT'}))
    # Successful execution may reject a model; independent experiments may still follow.
    assert completed_dependency(plan)==sha(report)
    source.write_text('changed\n')
    with pytest.raises(ValueError,match='binding changed'):check_sources(plan)
