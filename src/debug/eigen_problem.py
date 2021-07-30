from petsc4py import PETSc
import fenics as fe
import dolfin_adjoint as da
import mshr
import glob
import ufl
import numpy as np
from scipy.sparse.linalg import eigs
from . import arguments


def eigen_test():

    A = PETSc.Mat().create()
    A.setSizes([5, 5])
    A.setType("aij")
    A.setUp()

    # First arg is list of row indices, second list of column indices
    A.setValues([1,2,3], [1,2,3], np.ones((3, 3)))
    A.assemble()

    # A = A.convert("dense")
    # A.getDenseArray()
 
    A = fe.PETScMatrix(A)
    print(A.array()) 

    eigvals, eigvecs = eigs(A.array(), 5, which='SM')

    print(eigvals.real)
    print(eigvecs.real)

    solver = fe.SLEPcEigenSolver(A)
    solver.parameters["solver"] = "krylov-schur"
    # solver.parameters["spectrum"] = "smallest magnitude"
    # solver.parameters["problem_type"] = "gen_hermitian"
    solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"
    solver.parameters["spectral_shift"] = 2.99


    raw_num_eigs = 5

    solver.solve(raw_num_eigs)
        
    eigen_vals = []
    eigen_vecs = []
    for i in range(raw_num_eigs):
        r, c, rx, cx = solver.get_eigenpair(i)
        eigen_vals.append(r)
        eigen_vecs.append(rx.get_local())

    eigen_vecs = np.array(eigen_vecs).T


    print(eigen_vals)
    print(eigen_vecs)


if __name__ == '__main__':
    eigen_test()
