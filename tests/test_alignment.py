from geneweaver.alignment import AlignmentResult, validate_comparison_inputs
from geneweaver.parser import FastaInputError
import pytest


def test_alignment_result_reports_mismatch_count() -> None:
    result = AlignmentResult((1, 4), reference_length=5, query_length=5)

    assert result.mismatch_count == 2


def test_comparison_inputs_are_normalized() -> None:
    assert validate_comparison_inputs("acgt", "ACGT") == ("ACGT", "ACGT")


def test_comparison_inputs_reject_invalid_characters() -> None:
    with pytest.raises(FastaInputError):
        validate_comparison_inputs("ACGT", "ACGX")
