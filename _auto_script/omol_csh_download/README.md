# OMol_CSH download on SC26

This helper downloads the three published closed-shell Hamiltonian HDF5 files
directly from Meta's public file host to the SC26 shared dataset volume:

`/dataset/seongsu/shared-home/datasets/omol_csh`

The exact HTTP content lengths are 276,516,996,009 bytes for train,
33,350,917,699 bytes for the 5k test set, and 8,203,349,480 bytes for the 1k
common test set. The helper downloads all three in parallel, retains resumable
`.part` files, rejects incorrect final sizes, and then opens each HDF5 file to
verify 57,559, 4,986, and 1,008 molecule groups containing `coords`,
`elements`, and `fock` respectively.

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_csh_download/download_omol_csh.sh status
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_csh_download/download_omol_csh.sh download
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_csh_download/download_omol_csh.sh verify
```

Runtime logs are stored under:

`/dataset/seongsu/shared-home/workspace/project/outputs/omol-csh-download`
