"""Pure-Python CPU implementation of positional DNA comparison."""

from time import perf_counter

from geneweaver.alignment import AlignmentResult, validate_comparison_inputs
from geneweaver.parser import split_sequence


def compare_cpu(
    reference: str, query: str, *, chunk_size: int = 4096
) -> tuple[AlignmentResult, float]:
    """Compare DNA sequences on the CPU and return the result and elapsed seconds.

    Positions are zero-based. Every differing paired base is a mismatch, and
    bases beyond the shorter sequence are reported at their sequence positions.
    """
    reference, query = validate_comparison_inputs(reference, query)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    mismatch_positions: list[int] = []
    paired_length = min(len(reference), len(query))
    started = perf_counter()

    for reference_chunk in split_sequence(reference[:paired_length], chunk_size):
        query_chunk = query[reference_chunk.start : reference_chunk.end]
        for offset, (reference_base, query_base) in enumerate(
            zip(reference_chunk.sequence, query_chunk)
        ):
            if reference_base != query_base:
                mismatch_positions.append(reference_chunk.start + offset)

    mismatch_positions.extend(range(paired_length, max(len(reference), len(query))))
    elapsed = perf_counter() - started

    return (
        AlignmentResult(
            mismatch_positions=tuple(mismatch_positions),
            reference_length=len(reference),
            query_length=len(query),
        ),
        elapsed,
    )
