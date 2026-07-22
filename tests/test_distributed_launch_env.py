import pytest

from maloq.train_utils.utils_compute import distributed_context


def test_distributed_context_reads_torchrun_environment():
    env = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}

    assert distributed_context(env) == (1, 2, 1)


def test_distributed_context_reads_openmpi_environment():
    env = {
        "OMPI_COMM_WORLD_RANK": "3",
        "OMPI_COMM_WORLD_SIZE": "8",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
    }

    assert distributed_context(env) == (3, 8, 1)


def test_distributed_context_preserves_slurm_fallback():
    env = {
        "SLURM_PROCID": "5",
        "SLURM_NTASKS": "16",
        "SLURM_LOCALID": "1",
    }

    assert distributed_context(env) == (5, 16, 1)


def test_distributed_context_rejects_partial_torchrun_environment():
    with pytest.raises(ValueError, match="RANK and WORLD_SIZE"):
        distributed_context({"RANK": "0"})
