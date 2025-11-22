import numpy as np
import matplotlib.pyplot as plt
from fock_utils import utils_orca_out, utils_tensor_decomp, fock_targets, basis_sets

data = np.load('/checkpoint/ocp/manasakani/dimers/H_H_dimers/H_H_7.7_0_1/density_mat.npz')
orca_output_filepath = '/checkpoint/ocp/manasakani/dimers/H_H_dimers/H_H_7.7_0_1/orca.out'
fock_matrices, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath, unrestricted=True)

print(data.files)

Ptotal = data['orca.scfp']
Pspin = data['orca.scfr']
alpha_fock = fock_matrices['alpha']
beta_fock = fock_matrices['beta']

print(Ptotal.shape)

n = int((np.sqrt(8 * len(Ptotal) + 1) - 1) // 2)
mat = np.zeros((n,n))
mat[np.triu_indices(n)] = Ptotal
mat = mat + mat.T - np.diag(mat.diagonal())

plt.imshow(np.log(np.abs(mat)))
plt.savefig("Ptotal.png", dpi=300)


n = int((np.sqrt(8 * len(Pspin) + 1) - 1) // 2)
mat = np.zeros((n,n))
mat[np.triu_indices(n)] = Pspin
mat = mat + mat.T - np.diag(mat.diagonal())



plt.imshow(np.log(np.abs(mat)))
plt.savefig("Pspin.png", dpi=300)

plt.imshow(np.log(np.abs(alpha_fock)))
plt.savefig("alphafock.png", dpi=300)


plt.imshow(np.log(np.abs(beta_fock)))
plt.savefig("betafock.png", dpi=300)


# scfp is A+B
# scfr is A-B
