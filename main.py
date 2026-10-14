"""Command-line entry point for the GeneWeaver CPU baseline."""

import argparse
from pathlib import Path

from geneweaver.benchmark import format_seconds
from geneweaver.cpu_engine import compare_cpu
from geneweaver.dashboard import GeneWeaverApp
from geneweaver.gpu_engine import GPUUnavailableError, compare_gpu, cuda_available
from geneweaver.parser import FastaInputError, parse_fasta_file


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare query DNA sequences against a reference FASTA file."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("sample_data/reference.fasta"),
        help="Reference FASTA path.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("sample_data/queries.fasta"),
        help="Query FASTA path.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=4096, help="Bases per CPU comparison chunk."
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Execution backend; unavailable CUDA falls back to CPU.",
    )
    parser.add_argument(
        "--ui", action="store_true", help="Open the interactive Textual dashboard."
    )
    return parser


def main() -> int:
    """Load FASTA files, compare every query, and display the CPU baseline."""
    args = build_parser().parse_args()
    if args.ui:
        GeneWeaverApp().run()
        return 0

    try:
        reference_records = parse_fasta_file(args.reference)
        query_records = parse_fasta_file(args.queries)
        if len(reference_records) != 1:
            raise FastaInputError(
                f"Reference FASTA must contain exactly one record; "
                f"found {len(reference_records)}."
            )

        reference = reference_records[0]
        print(f"Reference: {reference.identifier} ({len(reference.sequence)} bases)")
        print(f"Queries: {len(query_records)}")
        selected_mode = args.mode
        if selected_mode == "auto":
            selected_mode = "gpu" if cuda_available() else "cpu"
            if selected_mode == "cpu":
                print("CUDA unavailable; using CPU fallback.")
        elif selected_mode == "gpu" and not cuda_available():
            print("CUDA unavailable; using CPU fallback.")
            selected_mode = "cpu"
        print(f"Execution mode: {selected_mode.upper()}")

        for query in query_records:
            if selected_mode == "gpu":
                result, elapsed = compare_gpu(
                    reference.sequence, query.sequence, chunk_size=args.chunk_size
                )
            else:
                result, elapsed = compare_cpu(
                    reference.sequence, query.sequence, chunk_size=args.chunk_size
                )
            positions = ", ".join(map(str, result.mismatch_positions)) or "none"
            print(
                f"{query.identifier}: length={result.query_length}, "
                f"mismatches={result.mismatch_count}, "
                f"positions=[{positions}], engine={selected_mode.upper()}, "
                f"time={format_seconds(elapsed)}"
            )
    except (FastaInputError, GPUUnavailableError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
