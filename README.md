# GeneWeaver

GeneWeaver is an educational DNA-sequence comparison prototype. It reads
synthetic FASTA records, checks DNA symbols, compares each query to a reference
with a pure-Python CPU baseline, and reports the measured elapsed time. It is
not a clinically validated CRISPR off-target prediction system.

## Project structure

```text
geneweaver/
├── main.py
├── requirements.txt
├── README.md
├── sample_data/
│   ├── reference.fasta
│   └── queries.fasta
├── geneweaver/
│   ├── __init__.py
│   ├── parser.py
│   ├── alignment.py
│   ├── cpu_engine.py
│   ├── gpu_engine.py
│   ├── benchmark.py
│   └── dashboard.py
└── tests/
    ├── test_parser.py
    ├── test_alignment.py
    ├── test_cpu_engine.py
    ├── test_gpu_engine.py
    └── test_dashboard.py
```

### Modules and functions

- `parser.py`: `parse_fasta_file` uses Biopython to read records;
  `validate_sequence` normalizes and validates DNA; `split_sequence` yields
  chunks with zero-based start and exclusive end offsets.
- `alignment.py`: `AlignmentResult` stores mismatch coordinates and lengths;
  `validate_comparison_inputs` validates a reference/query pair.
- `cpu_engine.py`: `compare_cpu` performs the pure-Python comparison across
  chunks and returns the result plus elapsed seconds from `time.perf_counter()`.
- `gpu_engine.py`: a Numba `@cuda.jit` kernel compares ASCII-encoded DNA arrays.
  `compare_gpu` bounds device memory use by processing chunks and measures
  transfers, kernel execution, and copying results back. `cuda_available`
  checks for a usable CUDA device. The program never reports GPU mode unless
  that backend is actually used.
- `benchmark.py`: `format_seconds` formats a measured duration.
- `dashboard.py`: the Textual UI accepts reference/query FASTA paths and an
  execution mode, then displays progress, chunk counts, mismatches, and timing.
- `main.py`: provides the command-line CPU/GPU/automatic modes and opens the UI.

Allowed DNA symbols are `A`, `C`, `G`, `T`, and `N` (unknown base). Whitespace
is ignored and lowercase is normalized to uppercase. Empty sequences and
unsupported symbols are errors. Positions are zero-based; when sequence lengths
differ, every unpaired trailing base is counted as a mismatch at its position.

## Install and run on Windows PowerShell

From the project root, create and activate a virtual environment, then install
the requirements. This includes the optional CUDA and UI dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the CPU baseline against the included synthetic data:

```powershell
python main.py --reference sample_data\reference.fasta --queries sample_data\queries.fasta
```

The default is CPU mode. Choose another chunk size with `--chunk-size 4`. Use
`--mode gpu` to request CUDA or `--mode auto` to choose GPU when available and
CPU otherwise. An explicit GPU request also falls back to CPU when CUDA is not
available, and prints that fallback. GPU runtime or memory errors are reported
instead of being presented as successful GPU work.

Open the interactive dashboard:

```powershell
python main.py --ui
```

The dashboard starts with the included sample paths; edit either path to use
other FASTA files. Choose Automatic, CPU, or GPU, then select Compare. From
VS Code, open the project folder, select the `.venv` interpreter, and run these
commands in the integrated PowerShell terminal.

Run the Week 1 tests:

```powershell
python -m pytest
```

## Sample terminal output

```text
Reference: synthetic_reference (16 bases)
Queries: 4
Execution mode: CPU
exact_match: length=16, mismatches=0, positions=[none], engine=CPU, time=<measured> s
two_substitutions: length=16, mismatches=2, positions=[4, 15], engine=CPU, time=<measured> s
short_query: length=8, mismatches=8, positions=[8, 9, 10, 11, 12, 13, 14, 15], engine=CPU, time=<measured> s
long_query: length=18, mismatches=2, positions=[16, 17], engine=CPU, time=<measured> s
```

Timings depend on the machine and Python runtime; the CPU implementation is a
baseline, not a promised speedup.

## CUDA support and limitations

The CUDA backend requires a supported NVIDIA GPU, a working NVIDIA driver, and
a Numba/CUDA combination supported by that device. Installing Python packages
alone does not provide a GPU or its driver. On systems without a usable CUDA
device, the command-line runner and dashboard clearly announce CPU fallback.
GPU parity tests are skipped when CUDA is unavailable; passing those tests on
CUDA hardware is required before claiming the kernel ran on a GPU. This
prototype compares bases at matching positions only; it does not perform
gapped alignment, PAM recognition, or clinically validated off-target
prediction. Dask is installed for future distributed processing but is not
used by the current single-machine comparison pipeline.
