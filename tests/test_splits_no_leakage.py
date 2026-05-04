import pytest

from esm_probe.data.validation import assert_no_overlap


def test_overlap_detection() -> None:
    with pytest.raises(ValueError, match="Leakage detected"):
        assert_no_overlap(["a", "b"], ["b", "c"], "external")


def test_no_overlap() -> None:
    assert_no_overlap(["a"], ["b"], "external")
