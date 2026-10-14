import pytest

from geneweaver.cpu_engine import compare_cpu
from geneweaver.gpu_engine import compare_gpu, cuda_available


@pytest.mark.skipif(not cuda_available(), reason="CUDA hardware/driver is unavailable")
@pytest.mark.parametrize(
    ("reference", "query"),
    [
        ("ACGTAC", "ACGTAC"),
        ("ACGTAC", "ATGTAA"),
        ("ACGT", "AC"),
        ("AC", "ACGT"),
    ],
)
def test_gpu_results_match_cpu_results(reference: str, query: str) -> None:
    cpu_result, _ = compare_cpu(reference, query, chunk_size=2)
    gpu_result, _ = compare_gpu(reference, query, chunk_size=2)

    assert gpu_result == cpu_result


def test_gpu_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        compare_gpu("ACGT", "ACGT", chunk_size=0)
