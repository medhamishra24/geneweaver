import pytest

from geneweaver.cpu_engine import compare_cpu


def test_exact_match_has_no_mismatches() -> None:
    result, elapsed = compare_cpu("ACGT", "ACGT", chunk_size=2)

    assert result.mismatch_positions == ()
    assert result.mismatch_count == 0
    assert elapsed >= 0


def test_mismatch_positions_are_zero_based_across_chunks() -> None:
    result, _ = compare_cpu("ACGTACGT", "ATGTTCGA", chunk_size=3)

    assert result.mismatch_positions == (1, 4, 7)


@pytest.mark.parametrize(
    ("reference", "query", "expected"),
    [
        ("ACGT", "AC", (2, 3)),
        ("AC", "ACGT", (2, 3)),
        ("ACGT", "AGGT", (1,)),
    ],
)
def test_unequal_lengths_include_unpaired_positions(
    reference: str, query: str, expected: tuple[int, ...]
) -> None:
    result, _ = compare_cpu(reference, query)

    assert result.mismatch_positions == expected


def test_cpu_engine_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        compare_cpu("ACGT", "ACGT", chunk_size=0)
