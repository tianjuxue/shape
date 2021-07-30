import fenics as fe
import dolfin_adjoint as da
import scipy
import numpy as np
import matplotlib.pyplot as plt
from . import arguments
from .pdeco import RVE


class Buckling(RVE):
    def __init__(self, domain, case, mode, problem):
        super(Buckling, self).__init__(domain, case, mode, problem)


    def compute_objective(self):
        compress = -0.04
        H = fe.as_matrix([[compress, 0.], [0., compress]])
        self.RVE_solve(H, dr=False)

        dummy_l = fe.dot(da.Constant((0., 0.)), self.v) * fe.dx(domain=self.mesh)

        A = fe.PETScMatrix()
        b = fe.PETScVector()
        # fe.assemble(self.jacE, tensor=A)   
        fe.assemble_system(self.jacE, dummy_l, self.bcs, A_tensor=A, b_tensor=b)

        solver = fe.SLEPcEigenSolver(A)
        solver.parameters["solver"] = "krylov-schur"
        solver.parameters["spectrum"] = "target magnitude"
        solver.parameters["spectral_transform"] = "shift-and-invert"
        solver.parameters["spectral_shift"] = -1.

        solver.solve(32)
        
        eigen_vals = []
        eigen_vecs = []
        for i in range(8):
            r, c, rx, cx = solver.get_eigenpair(i)
            eigen_vals.append(r)
            eigen_vecs.append(rx)

        print(eigen_vals)
        plt.plot(eigen_vals, marker='o', color='black')
        plt.show()
        # print(eigen_vecs[2].get_local())

        vtkfile_mesh_tmp = fe.File(f'data/pvd/{self.case}/{self.mode}/{self.problem}/eigen.pvd')

        eigen_u = fe.Function(self.V)
        eigen_u.vector()[:] = eigen_vecs[0]
        eigen_u.rename("e", "e")
        vtkfile_mesh_tmp << eigen_u

        return float(self.J)


    def forward_runs(self):
    	pass


def main():
    pde = Buckling(domain='rve', case='buckling', mode='critical', problem='inverse')    
    pde.run()


if __name__ == '__main__':
    main()
