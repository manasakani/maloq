# OMol25 open-shell metal-organic download

This helper restores only the strict metal-organic `ml_mo` rows with
`multiplicity > 1`. It downloads the untouched density archive and the ORCA
archive used to recover the Hamiltonian:

- `density_mat.npz`: fp64 `orca.scfp` and `orca.scfr`
- `orca.tar.zst`: `orca.out` with sequential alpha/beta UHF Fock matrices

It never writes into the repacked `electronic/ml_mo` tree. The default
destination is:

```text
/home1/irteam/data-vol1/data/omol25/open_shell_restore/ml_mo/...
```

## Why the raw ORCA archive is retained

The existing Quasar `convert_fock.py` stores one `fock_packed` array. In an
unrestricted ORCA output the alpha and beta matrices follow one another below
one `FOCK` header, so that converter can overwrite the first matrix with the
second. Do not run it on this restore tree. This helper verifies that two
distinct complete Fock matrices are present and keeps the source archive.

## Quasar commands

Run from the Quasar environment that owns the existing Globus login and
destination endpoint:

```bash
python download_omol_open_shell.py --list
python download_omol_open_shell.py --preflight
python download_omol_open_shell.py --all --parallel 4 --batch-size 500
python download_omol_open_shell.py --status
python download_omol_open_shell.py --verify-all
```

The selection contains 30,815 samples and 61,630 transferred files. Globus
uses checksum synchronization and checksum verification. Batch definitions,
task IDs, retry state, and validation reports are stored below
`open_shell_restore/_transfer/`.

The preflight is deterministic and covers multiplicities 2 through 6 and 3d,
4d, and 5d transition-metal periods while preferring small AO systems.
