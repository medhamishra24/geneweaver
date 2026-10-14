"""Optional Numba CUDA implementation of positional DNA comparison."""

from time import perf_counter

import numpy as np
from numba import cuda
from numba.cuda.cudadrv import error as cuda_error

from geneweaver.alignment import AlignmentResult, validate_comparison_inputs

THREADS_PER_BLOCK = 256
CUDA_ERRORS = (cuda_error.CudaSupportError,) + tuple(
    error_type
    for error_type in (
        getattr(cuda_error, "CudaDriverError", None),
        getattr(cuda_error, "CudaRuntimeError", None),
    )
    if isinstance(error_type, type) and issubclass(error_type, Exception)
)


class GPUUnavailableError(RuntimeError):
    """Raised when CUDA cannot be initialized on this computer."""


@cuda.jit
def _compare_kernel(reference, query, mismatch_flags):
    position = cuda.grid(1)
    if position < mismatch_flags.size:
        if position >= reference.size or position >= query.size:
            mismatch_flags[position] = 1
        elif reference[position] != query[position]:
            mismatch_flags[position] = 1
        else:
            mismatch_flags[position] = 0


def cuda_available() -> bool:
    """Return whether Numba can see a usable CUDA device."""
    try:
        return cuda.is_available()
    except (OSError,) + CUDA_ERRORS:
        return False


def compare_gpu(
    reference: str, query: str, *, chunk_size: int = 1_000_000
) -> tuple[AlignmentResult, float]:
    """Compare two sequences on CUDA, including transfer and result-copy time.

    Work is bounded by ``chunk_size`` so long sequences do not require one
    device allocation proportional to their full length.
    """
    reference, query = validate_comparison_inputs(reference, query)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not cuda_available():
        raise GPUUnavailableError(
            "CUDA is unavailable. Install a compatible NVIDIA driver and "
            "CUDA-enabled Numba environment, or select CPU mode."
        )

    mismatch_positions: list[int] = []
    maximum_length = max(len(reference), len(query))
    started = perf_counter()

    try:
        for start in range(0, maximum_length, chunk_size):
            end = min(start + chunk_size, maximum_length)
            reference_part = np.frombuffer(
                reference[start:end].encode("ascii"), dtype=np.uint8
            )
            query_part = np.frombuffer(query[start:end].encode("ascii"), dtype=np.uint8)
            chunk_length = end - start
            device_reference = cuda.to_device(reference_part)
            device_query = cuda.to_device(query_part)
            device_flags = cuda.device_array(chunk_length, dtype=np.int32)
            blocks = (chunk_length + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

            _compare_kernel[blocks, THREADS_PER_BLOCK](
                device_reference, device_query, device_flags
            )
            flags = device_flags.copy_to_host()
            mismatch_positions.extend(
                start + int(position) for position in np.flatnonzero(flags)
            )
    except (MemoryError,) + CUDA_ERRORS as error:
        raise RuntimeError(f"CUDA comparison failed: {error}") from error

    elapsed = perf_counter() - started
    return (
        AlignmentResult(
            mismatch_positions=tuple(mismatch_positions),
            reference_length=len(reference),
            query_length=len(query),
        ),
        elapsed,
    )
