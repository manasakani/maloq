# MALOQ - Massively Accelerated Learning of Operators for Quantum-transport

<p align="center">
  <img src="/images/maloq.png" alt="MALOQ Logo" width="150" />
</p>

<p align="center">
  <strong>Orbital interactions at scale.</strong>
</p>

---

## 📌 About
**MALOQ** is a scalable, equivariant Graph Neural Network (GNN) framework designed for the rapid prediction of Hamiltonian matrices and quantum operators. By leveraging $SO(2)$-equivariant kernels, symmetry reductions, and efficient data processing pipelines, MALOQ learns structure-property mappings to complex orbital interactions across diverse chemical spaces.

---

## 📊 Supported Datasets
The model is pre-configured to work with three molecular Hamiltonian matrix datasets. Ensure you download the datasets and update the `dbpath` in the respective config files. In addition, it's possible to set up your own materials dataset, using the `cp2k_material' option. This option is not yet documented, and there are a lot of small details to iron out in the data generation process. We are in the process of documenting how to generate materials Hamiltonian data to train models that can generalize to large-scale material structures. 

| Dataset | Type | Download Link | Config File (in /examples) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **MD17/QM7** | Small Molecules | [Website](https://quantum-machine.org/datasets/) | `run_QM7.py` | Water and other small molecules. |
| **nablaDFT** | DFT Benchmarks | [GitHub](https://github.com/AIRI-Institute/nablaDFT) | `run_nablaDFT.py` | Druglike molecules (30-50 atoms) with heavy atoms. |
| **OMol_CSH_58k** | Large Molecules | [HF](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md) | `run_omol_csh.py` | Large molecules (10-150 atoms). |

---

## 👥 Code contributors

* **(MALOQ)** – Manasa Kaniselvan, Denghui Lu, Alexander Maeder, Alexandros Nikolaos Ziogas, Mathieu Luisier
* **(HELM)** – Manasa Kaniselvan and Daniel Levine
---

## 📜 Citation

If you use **MALOQ** in your research or find the framework helpful, please include the following two citations:

### Data processing acceleration and distribution (MALOQ)
```bibtex
@misc{maloq,
  title  = {MALOQ: Massively Accelerated Learning of Operators for Quantum-transport},
  author = {Kaniselvan, Manasa and Maeder, Alexander and Lu, Denghui and Ziogas, Alexandros Nikolaos and Luisier, Mathieu},
  year   = {2026},
  doi    = {10.48550/arXiv.2606.28911},
  url    = {https://arxiv.org/abs/2606.28911}
}
```

### Core adapted architecture for electronic structure learning (HELM)
```bibtex
@misc{helm,
  title     = {Learning from the electronic structure of molecules across the periodic table},
  author    = {Kaniselvan, Manasa and Miller, Benjamin Kurt and Gao, Meng and Nam, Juno and Levine, Daniel S.},
  publisher = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  doi       = {10.48550/ARXIV.2510.00224},
  url       = {https://arxiv.org/abs/2510.00224},
  primaryClass = {physics.chem-ph}
}
```
