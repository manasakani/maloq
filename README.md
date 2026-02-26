# MALOQ - Massively Accelerated Learning of Operators for Quantum-transport

<p align="center">
  <img src="/images/maloq.png" alt="MALOQ Logo" width="150" />
</p>

<p align="center">
  <strong>Orbital interactions at scale.</strong>
</p>

---

## 📌 About
**MALOQ** is a scalable, equivariant Graph Neural Network (GNN) framework designed for the rapid prediction of Hamiltonian matrices and quantum operators. By leveraging $SO(3)$ equivariance and efficient message-passing, MALOQ captures complex orbital interactions across diverse chemical spaces.

This repository is the official implementation for our submission to **ICLR 2026**.

---

## 📊 Supported Datasets
The model is pre-configured to work with three major Hamitlonian matrix datasets. Ensure you download the datasets and update the `dbpath` in the respective config files.

| Dataset | Type | Config Script | Description |
| :--- | :--- | :--- | :--- |
| **QM7** | Small Molecules | `run_QM7.py` | Water and other small molecules. |
| **nablaDFT** | DFT Benchmarks | `run_nablaDFT.py` | Druglike molecules with 30-50 atoms each, containing a few heavy atoms |
| **Omol_CSH_58k** | Large Scale | `run_omol_csh.py` | 58k samples focused on large molecules (10-150 atoms) and complex orbital systems. |

---

## Getting Started

Clone the repository and install the required dependencies (see requirements.txt).

git clone [https://github.com/your-username/maloq.git](https://github.com/your-username/maloq.git)
cd maloq
pip install -r requirements.txt

python run_QM7.py or python run_nablaDFT.py or python run_omol_csh.py