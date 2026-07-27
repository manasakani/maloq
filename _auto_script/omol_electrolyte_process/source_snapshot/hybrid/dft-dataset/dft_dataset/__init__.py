import sys as _sys
from pathlib import Path as _Path

# bare import 호환 (from molecule import ... 등): src/ 를 sys.path에 추가
_src_dir = str(_Path(__file__).parent)
if _src_dir not in _sys.path:
    _sys.path.insert(0, _src_dir)

"""dft-dataset: Unified DFT data toolkit for MLIP / MLDFT / density prediction.

Core:
    Molecule, BasisInfo          — 통합 데이터 구조
    decode_fields                — 필드 선택적 디코딩 (fast path)

Data loading:
    LMDBDataset                  — LMDB 읽기/쓰기 (lazy env, field-selective)
    DFTCollator, GraphOptions    — 배칭 + OTF graph + PBC + rename
    BucketBatchSampler           — nao 기준 버킷 배칭
    build_dataloader             — 원라이너 DataLoader 팩토리

Storage:
    PickleFormat, SplitKeyFormat — 교체 가능한 직렬화 포맷

Prediction:
    PredictionResult             — 예측 저장/로드/분석
    PredictionCollector          — eval loop에서 batch 수집
    ComparisonReport             — reference 비교 통계

ASE integration:
    MLIPCalculator               — ASE Calculator (torch/pyg/jax/pyscf)
    MLDFTCalculator              — Hamiltonian prediction → SCF
"""

from .molecule import Molecule, BasisInfo, decode_fields
from .lmdb_dataset import LMDBDataset
from .collate import (
    DFTCollator,
    GraphOptions,
    BlockOptions,
    BucketBatchSampler,
    MixDataset,
    MixSampler,
    build_dataloader,
    build_mix_dataloader,
    PRESETS,
    to_pyg_data,
    to_pyg_batch,
)
from .storage_formats import (
    StorageFormat,
    PickleFormat,
    SplitKeyFormat,
    get_format,
    FORMAT_REGISTRY,
)
from .prediction import (
    PredictionResult,
    PredictionCollector,
    ComparisonReport,
)
from .calculator import (
    MLIPCalculator,
    MLDFTCalculator,
    MolecularDynamics,
    MDLogger,
    OMol25Evaluator,
    optimize,
)
