import json
import numpy as np
import scipy
from pyscf import gto, dft,scf
import re
from collections import defaultdict
import periodictable as pt
import scipy as sp
from pyscf import tools
from . import utils_orca_out

def get_permute_phase(mol:gto.Mole, elt_reorder:dict[str, list], elt_phase:dict[str, list])-> tuple[list, list]:
    """
    Determine permutation and phasing vectors for the given molecule

    :param mol: PySCF molecule of interest
    :param elt_reorder: tabulated permuations for each element (keys are
                    element abbreviations, values are permuation as one list)
    :param elt_phase: tabulated phases for each element (keys are element
                    abbreviations, values are phases (+/-1) as one list)
    :return: Vectors which will permute and phase structures to align ORCA
             ordering with PySCF ordering
    """
    atom_list = [atom[0] for atom in mol._atom]
    permutation = []
    phase = []
    curr_index = 0
    for atom in atom_list:
        if atom in elt_reorder:
            permutation.extend([curr_index + i for i in elt_reorder[atom]])
            phase.extend(elt_phase[atom])
            # print(f"Element {atom}: adding permutation { [curr_index + i for i in elt_reorder[atom]] } and phase { elt_phase[atom] }")
            curr_index += len(elt_reorder[atom])
        else:
            raise ValueError(f"Element {atom} not found in reorder mapping.")
    return permutation, phase

def permute_mat(mat:np.array, perm:list, phase:list, orca_to_pyscf:bool=True)->np.array:
    """
    Reorder (and adjust phases) on matrices to switch between ORCA and
    PySCF ordering

    :param mat: 2D numpy array (matrix) to be reordered
    :param perm: system-specific permutation vector
    :param phase: system-specific phase vector
    :bool orca_to_pyscf: If True, convert the matrix from ORCA order to
                         PySCF order. If False, go the other way.
    :return: reordered matrix
    """
    phase_array = np.array(phase)
    if orca_to_pyscf:
        # Reorder ORCA to PySCF ordering
        new_mat = mat[np.ix_(perm, perm)]
        new_mat = phase_array[:, None] * new_mat * phase_array[None, :]
    else:
        # Reorder PySCF to ORCA ordering
        new_mat = phase_array[:, None] * mat * phase_array[None, :]
        new_mat = new_mat[np.ix_(np.argsort(perm), np.argsort(perm))]
    return new_mat

def get_structure()->gto.Mole:
    """
    Get structure of system to compute energy. Subject to change
    once we decide how this looks.
    """
    # 1. Define molecule (example: H2O)
    water = '''
        O  0.000  0.000  0.000
        H  0.000  1.000  0.000
        H  0.000  0.000  1.000
        '''
    n2 = '''
        N  0.000  0.000  0.000
        N  0.000  0.000  1.100
        '''
    atom=n2
    mol = gto.M(atom = atom, basis='def2-tzvpd', ecp='def2-tzvpd')
    return mol

def build_density(mol:gto.Mole, F:np.array, S:np.array=None, save_density=False, name='0')->np.array:
    """
    Build the density by diagonalizing FC=SCe.

    Currently assumes that the system is closed-shell.

    :param mol: Molecule of interest (to provide number of electrons)
    :param F: Fock matrix
    :return: Density in the AO basis
    """
    # Get overlap matrix from PySCF to diagonalize F
    if S is None:
        print("Calculating overlap matrix from PySCF")
        S = mol.intor('int1e_ovlp')

    # filter small eigenvalues
    tol = 1e-1 # 1e-1
    s, U = sp.linalg.eigh(S)
    keep = s > tol
    s_kept = s[keep]
    U_kept = U[:, keep]

    print("Filtered out {} small eigenvalues of S".format(np.sum(~keep)))

    X = U_kept @ np.diag(1.0 / np.sqrt(s_kept))            # N_AO x r  (r = rank)
    Fp = X.T @ F @ X                                       # r x r
    eps, Cp = sp.linalg.eigh(Fp)                           # Cp: r x r
    C = X @ Cp                                             # N_AO x r
    # e, C = scipy.linalg.eigh(F, S) # remove!

    # Build density matrix (assuming closed-shell and N electrons)
    nelec = mol.nelectron
    print("Number of electrons: ", nelec)
    nocc = nelec // 2
    P = 2 * C[:, :nocc] @ C[:, :nocc].T

    # Save volumetric electron density
    if save_density:
        save_xyz(mol, 'structure_'+name+'.xyz')
        save_density_cube(mol, P, filename='density_'+name+'.cube')

    return P

def get_density_from_fock(atom_graph, Fock, HELM, label='0'):

    basis = HELM['basis']
    functional = HELM['functional']
    atomic_numbers = atom_graph.atomic_numbers
    positions = atom_graph.pos

    atom_list = []
    for z, pos in zip(atomic_numbers.numpy(), positions):
        element_symbol = utils_orca_out.periodic_table_number[z]
        atom_list.append([element_symbol, pos])

    # Create molecule
    mol = gto.M(
        atom=atom_list,
        basis=basis,
        unit='Angstrom'
    )

    F = utils_orca_out.sort_by_m(Fock, HELM['orbital_basis'], atomic_numbers.numpy(), direction="e3nn_to_pyscf")
    S = mol.intor('int1e_ovlp')

    # filter small eigenvalues
    tol = 1e-3 # 1e-1
    s, U = sp.linalg.eigh(S)
    keep = s > tol
    s_kept = s[keep]
    U_kept = U[:, keep]
    print("Filtered out {} small eigenvalues of S".format(np.sum(~keep)))

    X = U_kept @ np.diag(1.0 / np.sqrt(s_kept))            # N_AO x r  (r = rank)
    Fp = X.T @ F @ X                                       # r x r
    eps, Cp = sp.linalg.eigh(Fp)                           # Cp: r x r
    C = X @ Cp                                             # N_AO x r

    # Build density matrix (assuming closed-shell and N electrons)
    nelec = mol.nelectron
    nocc = nelec // 2
    P = 2 * C[:, :nocc] @ C[:, :nocc].T
    save_density_cube(mol, P, filename=f'density_{label}.cube')

def get_integrals(mol:gto.Mole,
                P:np.array, functional:str,
                dataset_name:str='omol'
                )->tuple[np.array, float, np.array]:
    """
    Get needed additional integrals/etc. for total energy evaluation.

    The integrals are Hcore and we also need the XC functional evaluated
    on a grid to provide E_xc and V_Xxc.

    :param mol: Molecule of interest (to provide grids)
    :param P: Density matrix in the AO basis
    :return: Core Hamiltonian (i.e. kinetic energy (T) + electron-nuclear
                attraction (V_ne) and any ECP corrections),
             Exchange-correlation energy (E_xc) (including VV10),
             Exchange-correlation potential (V_xc) (including VV10)
    """
    H = scf.hf.get_hcore(mol)

    grids = dft.gen_grid.Grids(mol)

    if dataset_name == 'nablaDFT':
        grids.atom_grid= (75,302)
    else:
        grids.atom_grid= (99,590)

    grids.prune = 'treutler'
    grids.build()

    ni = dft.numint.NumInt()
    elec, E_xc, V_xc = ni.nr_rks(mol, grids, functional, P)

    if functional == 'wb97m-v':
        nlcgrids = dft.gen_grid.Grids(mol)
        nlcgrids.atom_grid= (50,194)
        nlcgrids.prune = 'treutler'
        nlcgrids.build()
        elec2, E_nlc, V_nlc = ni.nr_nlc_vxc(mol, nlcgrids, 'wb97m-v', P)
        return H, E_xc + E_nlc, V_xc + V_nlc
    elif functional == 'pbe' or functional == 'wb97x-d':
        return H, E_xc, V_xc
    else:
        raise ValueError(f"Functional {functional} not implemented.")

def save_density_cube(mol, P, filename='density.cube',
                     grid_spacing=0.4,   # Angstroms per voxel
                     margin=3.0,         # Angstroms of padding around molecule
                     save_density=True
                    ):
    """
    Save the electron density to a cube file for visualization.
    :param mol: PySCF molecule object
    :param P: Density matrix (AO basis)
    :param filename: Output filename
    :param grid_spacing: Desired grid spacing in Angstroms
    :param margin: Padding (in Angstroms) to add around the molecule
    """

    # Get atomic coordinates in Angstroms
    coords = mol.atom_coords(unit='Angstrom')
    min_corner = coords.min(axis=0) - margin
    max_corner = coords.max(axis=0) + margin
    box_lengths = max_corner - min_corner

    # Calculate number of grid points along each axis
    nx, ny, nz = np.ceil(box_lengths / grid_spacing).astype(int)

    # # Generate grid points
    # xs = np.linspace(min_corner[0], max_corner[0], nx)
    # ys = np.linspace(min_corner[1], max_corner[1], ny)
    # zs = np.linspace(min_corner[2], max_corner[2], nz)
    # grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T

    # # Compute density at each grid point
    # ao = mol.eval_gto('GTOval_sph', grid)
    # rho = np.einsum('pi,ij,pj->p', ao, P, ao)
    # density_grid = rho.reshape((nx, ny, nz))

    # PySCF's built-in cube file writer
    if save_density:
        tools.cubegen.density(mol, filename, P, nx, ny, nz)
        print(f"Electron density cube file written to {filename}")

def save_xyz(mol, filename='structure.xyz'):
    """
    Save atomic coordinates as an .xyz file from a PySCF Mole object.
    """
    # mol._atom: list of (symbol, (x, y, z)) in Angstroms if input as Angstroms
    atoms = mol._atom
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated by from save_xyz\n")
        for sym, (x, y, z) in atoms:
            f.write(f"{sym:2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
    print(f"Wrote XYZ to {filename}")

def main():
    with open('element_perm.json', 'r') as fh:
        json_data = json.loads(fh.read())
    elt_reorder = json_data['element_permuations']
    elt_phase = json_data['element_phases']

    mol = get_structure()

    perm, phase = get_permute_phase(mol, elt_reorder, elt_phase)
    # Load Fock and reorder to PySCF ordering
    F = np.load('Fock_wb97mv.npy')
    F = permute_mat(F, perm, phase)

    # Get intermediate quantities
    P = build_density(mol, F)
    H, E_xc, V_xc = get_integrals(mol, P)

    # Compute energy
    E_nn = mol.energy_nuc()
    total_energy = 0.5 * np.einsum('ij,ji', P,  H + F - V_xc) + E_xc + E_nn

    print(total_energy)

if __name__ == '__main__':
    main()
