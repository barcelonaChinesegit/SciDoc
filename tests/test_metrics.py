import pytest

from evaluation.metrics import evidence_metrics


@pytest.mark.parametrize("p,g,expected", [
    ({1, 2}, {1, 2}, (1, 1, 1)), ({1, 2}, {2, 3}, (.5, .5, .5)),
    ({1, 2, 3}, {2}, (1 / 3, 1, .5)), (set(), set(), (1, 1, 1)),
    (set(), {2}, (0, 0, 0)), ({2}, set(), (0, 0, 0)), ({1}, {2}, (0, 0, 0))])
def test_evidence(p, g, expected):
    assert evidence_metrics(p, g) == pytest.approx(expected)
