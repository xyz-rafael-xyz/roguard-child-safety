from training.verify_native_runtime_v9 import audit


def test_abstention_counts_as_error_for_both_classes_and_strata():
    rows = [{'source': source} for source in ('formal', 'v7_training_frame_projection') for _ in range(2)]
    predictions = [{'gold': gold, 'predicted': None} for _ in range(2) for gold in (0, 1)]
    report = audit(rows, predictions)
    for stratum in report.values():
        assert stratum['missed'] == stratum['false_reviews'] == 1
        assert stratum['recall'] == 0
        assert stratum['false_review_rate'] == 1
        assert stratum['passes'] is False


def test_strict_integer_gates_do_not_round_small_stratum_failures_away():
    rows = [{}]*199
    predictions = [{'gold': 1, 'predicted': 1} for _ in range(99)]+[{'gold': 0, 'predicted': 0} for _ in range(100)]
    predictions[0]['predicted'] = 0
    report = audit(rows, predictions)
    assert report['formal']['recall'] > .989
    assert report['formal']['passes'] is False
    assert report['conversational']['passes'] is False
