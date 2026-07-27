"""Node-latent implicit operator-projection experiment."""

from .backbone import (
    OP_PROJECTION_ARCHITECTURE,
    OpProjectionBackbone,
)
from .loss import probe_matrix_mse, rademacher_probes, relative_action_error
from .operator import BoundOperatorCallback, bind_operator_callback
from .projection import (
    CoupledToPackedAO,
    OpProjectionHead,
    OpProjectionModel,
    PackedAOBlockMatvec,
)
from .training import (
    OpProjectionMatrixMetricsConfig,
    OpProjectionTrainingConfig,
    coupled_label_action,
    deterministic_probe_seed,
    exact_matrix_sample_indices,
    identity_column_ranges,
    matrix_column_error_sums,
    molecule_ao_bounds,
    molecule_probe_statistics,
    should_log_optimizer_step,
)

FEATURE_SLUG = "op_projection"

__all__ = [
    "BoundOperatorCallback",
    "CoupledToPackedAO",
    "FEATURE_SLUG",
    "OP_PROJECTION_ARCHITECTURE",
    "OpProjectionBackbone",
    "OpProjectionHead",
    "OpProjectionMatrixMetricsConfig",
    "OpProjectionModel",
    "OpProjectionTrainingConfig",
    "PackedAOBlockMatvec",
    "bind_operator_callback",
    "coupled_label_action",
    "deterministic_probe_seed",
    "exact_matrix_sample_indices",
    "identity_column_ranges",
    "matrix_column_error_sums",
    "molecule_ao_bounds",
    "molecule_probe_statistics",
    "should_log_optimizer_step",
    "probe_matrix_mse",
    "rademacher_probes",
    "relative_action_error",
]
