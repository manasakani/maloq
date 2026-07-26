# OMol25 open-shell metal-organic download

This helper restores only the strict metal-organic `ml_mo` rows with
`multiplicity > 1`. It downloads the untouched density archive and the ORCA
archive used to recover the Hamiltonian:

- `density_mat.npz`: fp64 `orca.scfp` and `orca.scfr`
- `orca.tar.zst`: `orca.out` with sequential alpha/beta UHF Fock matrices

It never writes into the repacked `electronic/ml_mo` tree. New transfers must
be initiated on SC26 and must use an SC26 Globus collection rooted under
`/dataset`. The default destination is:

```text
/dataset/seongsu/shared-home/datasets/omol25_open_shell_source/ml_mo/...
```

## Why the raw ORCA archive is retained

The existing Quasar `convert_fock.py` stores one `fock_packed` array. In an
unrestricted ORCA output the alpha and beta matrices follow one another below
one `FOCK` header, so that converter can overwrite the first matrix with the
second. Do not run it on this restore tree. This helper verifies that two
distinct complete Fock matrices are present and keeps the source archive.

## SC26 commands

Run on an SC26 server after installing Globus Connect Personal, exposing a
writable path under `/dataset/seongsu/shared-home/datasets`, and logging the
Globus CLI in. Set the SC26 collection UUID explicitly:

```bash
export OMOL_GLOBUS_DESTINATION_ENDPOINT=<SC26-collection-UUID>
python download_omol_open_shell.py --list
python download_omol_open_shell.py --preflight
python download_omol_open_shell.py --all --parallel 4 --batch-size 500
python download_omol_open_shell.py --status
python download_omol_open_shell.py --verify-all
```

The historical Quasar destination UUID is explicitly rejected. Quasar may be
used as a read-only source for already completed artifacts, but new download
jobs and their destination must be owned by SC26.

The selection contains 30,815 samples and 61,630 transferred files. Globus
uses checksum synchronization and checksum verification. Batch definitions,
task IDs, retry state, and validation reports are stored below
`omol25_open_shell_source/_transfer/`.

The preflight is deterministic and covers multiplicities 2 through 6 and 3d,
4d, and 5d transition-metal periods while preferring small AO systems.
