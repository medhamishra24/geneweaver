"""FASTA loading, DNA validation, and position-preserving chunking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from Bio import SeqIO

DNA_ALPHABET = frozenset("ACGTN")


class FastaInputError(ValueError):
    """Raised when a FASTA file is empty, unreadable, or contains invalid DNA."""


@dataclass(frozen=True)
class SequenceRecord:
    """A validated FASTA record with its identifier and normalized DNA."""

    identifier: str
    sequence: str


@dataclass(frozen=True)
class SequenceChunk:
    """A sequence slice whose end coordinate is exclusive."""

    start: int
    end: int
    sequence: str


def validate_sequence(sequence: str, *, identifier: str = "sequence") -> str:
    """Normalize a DNA string and reject empty strings or unsupported characters."""
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise FastaInputError(f"{identifier!r} has an empty DNA sequence.")

    invalid = sorted(set(normalized) - DNA_ALPHABET)
    if invalid:
        characters = ", ".join(repr(character) for character in invalid)
        raise FastaInputError(
            f"{identifier!r} contains invalid DNA character(s): {characters}. "
            "Allowed characters are A, C, G, T, and N."
        )
    return normalized


def parse_fasta_file(path: str | Path) -> list[SequenceRecord]:
    """Parse and validate every record in a FASTA file using Biopython."""
    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise FastaInputError(f"FASTA file does not exist: {fasta_path}")

    try:
        with fasta_path.open("r", encoding="utf-8") as handle:
            records = [
                SequenceRecord(
                    identifier=record.id,
                    sequence=validate_sequence(str(record.seq), identifier=record.id),
                )
                for record in SeqIO.parse(handle, "fasta")
            ]
    except (OSError, UnicodeError) as error:
        raise FastaInputError(f"Could not read FASTA file {fasta_path}: {error}") from error

    if not records:
        raise FastaInputError(f"FASTA file contains no sequence records: {fasta_path}")
    return records


def split_sequence(sequence: str, chunk_size: int) -> Iterator[SequenceChunk]:
    """Yield non-overlapping chunks with zero-based, half-open sequence positions."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    for start in range(0, len(sequence), chunk_size):
        end = min(start + chunk_size, len(sequence))
        yield SequenceChunk(start=start, end=end, sequence=sequence[start:end])
