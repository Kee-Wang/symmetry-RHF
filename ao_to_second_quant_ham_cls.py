from openfermion import transforms
from openfermion.chem.molecular_data import spinorb_from_spatial
import numpy as np
from openfermion import general_basis_change
import openfermion.ops.representations as reps
import qutip as qu


def ao_to_mo_second_quant_ham(E_nu, c1_ao, c2_ao, C):
    """
    Args:
        E_nu: nuclear repulsion energy
        c1_ao: 1-body core ham matrix in ao basis
        c2_ao: 2-body eri ham in ao basis
        C: optimized coefficient

    Returns:
        second quantized Ham

    """

    c2_ao = np.einsum('psqr', c2_ao) # Psi4 chemistry notation to general physics notation in second quant

    c1_mo = general_basis_change(c1_ao, C, (1, 0))
    c2_mo = general_basis_change(c2_ao, C, (1, 1, 0, 0))

    c1_so, c2_so = spinorb_from_spatial(c1_mo, c2_mo)

    molecular_hamiltonian = reps.InteractionOperator(
        E_nu, c1_so, 1 / 2 * c2_so)

    #
    of_second_ham2 = transforms.normal_ordered(molecular_hamiltonian)
    return of_second_ham2


single_qubit_pauli = {'I': qu.qeye(2),
                      'X': qu.sigmax(),
                      'Y': qu.sigmay(),
                      'Z': qu.sigmaz()}


def pauli_string_to_matrix(p):
    op_list = [single_qubit_pauli[s] for s in p]
    return qu.tensor(op_list).data




def qubit_operator_to_sparse_matrix(qubit_operator, nq):
    H_mat = 0
    for ps in qubit_operator:
        for key in ps.terms:
            ps_string = ['I'] * nq
            for pair in key:
                loc, pauli = pair
                ps_string[loc] = pauli

            coeff = ps.terms[key]
        ps_string = ''.join(ps_string)
        H_mat += coeff * pauli_string_to_matrix(ps_string)
    return H_mat