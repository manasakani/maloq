# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
import h5py
import json
import sys
import os

# Set your HDF5 file path here (or pass as first argument: python save_omol_csh_58k_keys.py path/to/file.h5)
dbpath = sys.argv[1] if len(sys.argv) > 1 else "omol_csh_1k_test_common.h5"
manifest_path = f"{dbpath}.keys.json"

if not os.path.exists(dbpath):
    print(f"Error: File '{dbpath}' not found!")
    sys.exit(1)

print(f"Building sidecar key index for {dbpath}...", flush=True)

keys = []

def find_molecule_groups(grp, prefix=''):
    """Recursively walks groups and stops as soon as a group containing 'fock' is found."""
    for k in grp:
        path = f"{prefix}/{k}" if prefix else k
        try:
            item = grp[k]
            if isinstance(item, h5py.Group):
                if 'fock' in item:
                    keys.append(path)
                else:
                    # Recurse deeper into sub-folders
                    find_molecule_groups(item, path)
        except Exception:
            continue

with h5py.File(dbpath, 'r') as f:
    find_molecule_groups(f)

print(f"Found {len(keys)} valid molecule keys.", flush=True)

with open(manifest_path, 'w') as f:
    json.dump(keys, f)

print(f"Done! Saved to {manifest_path}", flush=True)
