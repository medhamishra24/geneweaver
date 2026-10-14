"""Shared data structures and validation for sequence comparisons."""

from dataclasses import dataclass

from geneweaver.parser import validate_sequence


@dataclass(frozen=True)
class AlignmentResult:
    """Mismatch coordinates and sequence lengths for one comparison."""

    mismatch_positions: tuple[int, ...]
    reference_length: int
    query_length: int

    @property
    def mismatch_count(self) -> int:
        """Number of substitutions and unpaired trailing bases."""
        return len(self.mismatch_positions)


def validate_comparison_inputs(reference: str, query: str) -> tuple[str, str]:
    """Validate and normalize both sequences before an engine compares them."""
    return (
        validate_sequence(reference, identifier="reference"),
        validate_sequence(query, identifier="query"),
    )
