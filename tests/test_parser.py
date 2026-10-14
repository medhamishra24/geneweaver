from pathlib import Path

import pytest

from geneweaver.parser import (
    FastaInputError,
    parse_fasta_file,
    split_sequence,
    validate_sequence,
)


def test_parse_records_have_expected_values(tmp_path: Path) -> None:
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text(">one\nacgt\n>two\nNNaa\n", encoding="utf-8")

    records = parse_fasta_file(fasta)
    assert [(record.identifier, record.sequence) for record in records] == [
        ("one", "ACGT"),
        ("two", "NNAA"),
    ]


@pytest.mark.parametrize("sequence", ["", "ACGX", "ACGU"])
def test_validate_rejects_empty_or_invalid_dna(sequence: str) -> None:
    with pytest.raises(FastaInputError):
        validate_sequence(sequence)


def test_empty_fasta_is_rejected(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")

    with pytest.raises(FastaInputError, match="no sequence records"):
        parse_fasta_file(fasta)


def test_chunking_preserves_positions_and_final_partial_chunk() -> None:
    chunks = list(split_sequence("ACGTAC", 4))

    assert [(chunk.start, chunk.end, chunk.sequence) for chunk in chunks] == [
        (0, 4, "ACGT"),
        (4, 6, "AC"),
    ]
