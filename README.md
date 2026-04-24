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

This repository is the official implementation for our submission to **SC26**

---

## 📊 Supported Datasets
The model is pre-configured to work with three Hamiltonian matrix datasets. Ensure you download the datasets and update the `dbpath` in the respective config files.

| Dataset | Type | Download Link | Config File | Description |
| :--- | :--- | :--- | :--- | :--- |
| **MD17/QM7** | Small Molecules | [Website](https://quantum-machine.org/datasets/) | `run_QM7.py` | Water and other small molecules. |
| **nablaDFT** | DFT Benchmarks | [GitHub](https://github.com/AIRI-Institute/nablaDFT) | `run_nablaDFT.py` | Druglike molecules (30-50 atoms) with heavy atoms. |
| **OMol electronic structures** | Large Molecules | [Link](https://www.materialsdatafacility.org/spotlight/omol25#access-data) | `run_omol_csh.py` | Large molecules (10-150 atoms). |
| **Amorphous materials** | Material | [Link](https://huggingface.co/datasets/chexia8/Amorphous-Hamiltonians) | `run_cp2k_dataset.py` | Amorphous materials |

---

## 👥 Contributors

* **ICLR2026 (HELM)** – Manasa Kaniselvan and Daniel Levine
* **SC26 submission (MALOQ)** – Manasa Kaniselvan, Denghui Lu, Alexander Maeder, Alexandros Nikolaos Ziogas

---

## 📜 Citation

If you use **MALOQ** in your research or find the framework helpful, please cite the following papers:

### Data processing acceleration and distribution (MALOQ)
```bibtex
@misc{maloq,
  title  = {MALOQ: Massively Accelerated Learning of Operators for Quantum-transport},
  author = {TBD},
  year   = {2026},
  note   = {TBD},
  doi    = {X},
  url    = {X}
}
```

### Initial implementation for electronic structure learning (HELM)
```bibtex
@misc{helm,
  title     = {Learning from the electronic structure of molecules across the periodic table},
  author    = {Kaniselvan, Manasa and Miller, Benjamin Kurt and Gao, Meng and Nam, Juno and Levine, Daniel S.},
  publisher = {arXiv},
  year      = {2025},
  doi       = {10.48550/ARXIV.2510.00224},
  url       = {[https://arxiv.org/abs/2510.00224](https://arxiv.org/abs/2510.00224)},
  primaryClass = {physics.chem-ph}
}
```
