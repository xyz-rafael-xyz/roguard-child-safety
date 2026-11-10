"""Regression checks for the v9 fallback and strong out-of-scope override."""

import pytest

pytest.importorskip('numpy')

from roguard.gemma4_rbf_v10 import _binary_decision


def test_strong_negative_margin_clears_report_with_general_concern():
    assert _binary_decision(1, -1.0, 1.0) == 0
    assert _binary_decision(1, -0.61, 1.0) == 0


def test_ambiguous_margin_keeps_parent_report_with_personal_concern():
    assert _binary_decision(1, -0.45, 0.6) == 1


def test_agreed_negative_margins_clear_parent_false_alert():
    assert _binary_decision(1, -0.45, -0.6) == 0


def test_positive_override_requires_both_heads():
    assert _binary_decision(0, 0.6, 0.2) == 1
    assert _binary_decision(0, 0.6, -0.2) == 0
    assert _binary_decision(0, 0.4, 0.2) == 1
    assert _binary_decision(0, 0.39, 0.2) == 0


def test_language_threshold_can_clear_weak_concern_margin():
    assert _binary_decision(1, -0.4, 0.03, 0.1) == 0
