"""Prediction file I/O utilities.

Standard format for model prediction outputs:

    predictions/
    └── {model_name}/
        └── {dataset}/
            ├── metadata.yaml
            └── {mol_id}.npz

NPZ key convention:
    Required:
        hamiltonian    (nao, nao) for RKS, (2, nao, nao) for UKS
        mol_id         str
    Recommended:
        model_name     str
        unrestricted   bool
        basis          str  (e.g. "def2-svp")
        convention     str  (e.g. "pyscf", "psi4")
        energy_unit    str  ("Ha" or "eV")
        length_unit    str  ("Ang" or "Bohr")
    Optional:
        overlap        (nao, nao)
        orbital_energies  (nao,) or (2, nao)
        energy         float
        n_alpha        int
        n_beta         int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = {"hamiltonian", "mol_id"}
VALID_CONVENTIONS = {"pyscf", "psi4", "e3nn"}
VALID_ENERGY_UNITS = {"Ha", "eV"}
VALID_LENGTH_UNITS = {"Ang", "Bohr"}


def save_prediction(
    path: str | Path,
    hamiltonian: np.ndarray,
    mol_id: str,
    model_name: str = "unknown",
    *,
    unrestricted: bool | None = None,
    basis: str = "def2-svp",
    convention: str = "pyscf",
    energy_unit: str = "Ha",
    length_unit: str = "Ang",
    overlap: np.ndarray | None = None,
    orbital_energies: np.ndarray | None = None,
    energy: float | None = None,
    n_alpha: int | None = None,
    n_beta: int | None = None,
    **extra: Any,
) -> Path:
    """Save a prediction in standard NPZ format.

    Args:
        path: Output file path (.npz)
        hamiltonian: (nao, nao) for RKS or (2, nao, nao) for UKS
        mol_id: Molecule identifier
        model_name: Name of the model that produced this prediction
        unrestricted: Whether UKS. Auto-detected from hamiltonian shape if None.
        basis: Basis set name
        convention: Orbital convention ("pyscf", "psi4", "e3nn")
        energy_unit: Energy unit ("Ha" or "eV")
        length_unit: Length unit ("Ang" or "Bohr")
        overlap: Overlap matrix (nao, nao)
        orbital_energies: (nao,) for RKS or (2, nao) for UKS
        energy: Total energy (scalar)
        n_alpha: Number of alpha electrons (UKS)
        n_beta: Number of beta electrons (UKS)
        **extra: Additional arrays/scalars to store

    Returns:
        Path to saved file
    """
    path = Path(path)
    H = np.asarray(hamiltonian, dtype=np.float64)

    # Auto-detect unrestricted
    if unrestricted is None:
        unrestricted = H.ndim == 3

    # Validate
    if unrestricted:
        if H.ndim != 3 or H.shape[0] != 2:
            raise ValueError(
                f"UKS hamiltonian must be (2, nao, nao), got {H.shape}"
            )
        if H.shape[1] != H.shape[2]:
            raise ValueError(
                f"Hamiltonian must be square, got {H.shape[1]}x{H.shape[2]}"
            )
    else:
        if H.ndim != 2:
            raise ValueError(
                f"RKS hamiltonian must be (nao, nao), got {H.shape}"
            )
        if H.shape[0] != H.shape[1]:
            raise ValueError(
                f"Hamiltonian must be square, got {H.shape[0]}x{H.shape[1]}"
            )

    if convention not in VALID_CONVENTIONS:
        raise ValueError(
            f"convention must be one of {VALID_CONVENTIONS}, got '{convention}'"
        )

    data: dict[str, Any] = {
        "hamiltonian": H,
        "mol_id": np.array(mol_id),
        "model_name": np.array(model_name),
        "unrestricted": np.array(unrestricted),
        "basis": np.array(basis),
        "convention": np.array(convention),
        "energy_unit": np.array(energy_unit),
        "length_unit": np.array(length_unit),
    }

    if overlap is not None:
        data["overlap"] = np.asarray(overlap, dtype=np.float64)
    if orbital_energies is not None:
        data["orbital_energies"] = np.asarray(orbital_energies, dtype=np.float64)
    if energy is not None:
        data["energy"] = np.array(energy, dtype=np.float64)
    if n_alpha is not None:
        data["n_alpha"] = np.array(n_alpha, dtype=np.int64)
    if n_beta is not None:
        data["n_beta"] = np.array(n_beta, dtype=np.int64)

    for k, v in extra.items():
        data[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **data)
    return path


def load_prediction(path: str | Path) -> dict[str, Any]:
    """Load and validate a prediction file.

    Returns:
        Dict with at least 'hamiltonian', 'mol_id', 'unrestricted' keys.
        Arrays are numpy arrays, scalars are Python types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix == ".npz":
        return _load_npz(path)
    elif path.suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def _load_npz(path: Path) -> dict[str, Any]:
    raw = np.load(str(path), allow_pickle=True)

    if "hamiltonian" not in raw:
        raise ValueError(
            f"Missing 'hamiltonian' key. Available: {list(raw.keys())}"
        )

    H = raw["hamiltonian"]
    unrestricted = bool(raw["unrestricted"]) if "unrestricted" in raw else (H.ndim == 3)

    result: dict[str, Any] = {
        "hamiltonian": H,
        "unrestricted": unrestricted,
        "nao": H.shape[-1],
    }

    # Extract scalar/string fields
    for key in ["mol_id", "model_name", "basis", "convention",
                "energy_unit", "length_unit"]:
        if key in raw:
            val = raw[key]
            result[key] = str(val) if val.ndim == 0 else str(val)

    # Extract array fields
    for key in ["overlap", "orbital_energies"]:
        if key in raw:
            result[key] = raw[key]

    # Extract numeric scalars
    for key in ["energy", "n_alpha", "n_beta"]:
        if key in raw:
            val = raw[key]
            result[key] = float(val) if key == "energy" else int(val)

    return result


def _load_json(path: Path) -> dict[str, Any]:
    import json
    with open(path) as f:
        data = json.load(f)

    if "hamiltonian" not in data:
        raise ValueError(f"Missing 'hamiltonian' key in JSON")

    H = np.array(data["hamiltonian"], dtype=np.float64)
    unrestricted = data.get("unrestricted", H.ndim == 3)

    result: dict[str, Any] = {
        "hamiltonian": H,
        "unrestricted": unrestricted,
        "nao": H.shape[-1],
    }

    for key in ["mol_id", "model_name", "basis", "convention",
                "energy_unit", "length_unit"]:
        if key in data:
            result[key] = str(data[key])

    if "overlap" in data:
        result["overlap"] = np.array(data["overlap"], dtype=np.float64)
    if "orbital_energies" in data:
        result["orbital_energies"] = np.array(data["orbital_energies"], dtype=np.float64)
    if "energy" in data:
        result["energy"] = float(data["energy"])
    for key in ["n_alpha", "n_beta"]:
        if key in data:
            result[key] = int(data[key])

    return result


def validate_prediction(pred: dict, gt_nao: int | None = None) -> list[str]:
    """Validate a loaded prediction dict. Returns list of warnings (empty = OK)."""
    warnings = []

    H = pred.get("hamiltonian")
    if H is None:
        return ["Missing hamiltonian"]

    unrestricted = pred.get("unrestricted", False)

    if unrestricted:
        if H.ndim != 3 or H.shape[0] != 2:
            warnings.append(f"UKS hamiltonian shape invalid: {H.shape}")
        elif H.shape[1] != H.shape[2]:
            warnings.append(f"Hamiltonian not square: {H.shape}")
    else:
        if H.ndim != 2:
            warnings.append(f"RKS hamiltonian should be 2D, got {H.ndim}D")
        elif H.shape[0] != H.shape[1]:
            warnings.append(f"Hamiltonian not square: {H.shape}")

    if gt_nao is not None:
        pred_nao = H.shape[-1]
        if pred_nao != gt_nao:
            warnings.append(
                f"Dimension mismatch: GT is {gt_nao}x{gt_nao}, "
                f"prediction is {pred_nao}x{pred_nao}"
            )

    conv = pred.get("convention")
    if conv and conv not in VALID_CONVENTIONS:
        warnings.append(f"Unknown convention: {conv}")

    return warnings


def save_metadata(
    output_dir: str | Path,
    model_name: str,
    dataset: str,
    **kwargs: Any,
) -> Path:
    """Write metadata.yaml for a prediction directory."""
    import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_name": model_name,
        "dataset": dataset,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    meta.update(kwargs)

    path = output_dir / "metadata.yaml"
    with open(path, "w") as f:
        for k, v in meta.items():
            if isinstance(v, (list, dict)):
                import yaml
                yaml.dump({k: v}, f, default_flow_style=False)
            else:
                f.write(f"{k}: {v}\n")

    return path
