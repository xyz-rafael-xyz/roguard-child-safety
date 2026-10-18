from training.semantic_selection_v9 import selection_rank


def groups(passes, rate):
    return {kind: dict(passes=passes, false_review_rate=rate) for kind in ('formal', 'conversational')}


def test_semantic_quality_is_selected_only_after_accuracy_gates_pass():
    assert selection_rank(groups(True, .04), .21, 10, False) < selection_rank(groups(True, .02), .27, 0, False)
    assert selection_rank(groups(True, .04), .21, 10, False) < selection_rank(groups(False, .001), .01, 0, False)


def test_failed_candidates_still_prioritize_false_review_rate():
    assert selection_rank(groups(False, .1), .3, 10, False) < selection_rank(groups(False, .2), .1, 0, False)
