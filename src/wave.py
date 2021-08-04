import fenics as fe
import dolfin_adjoint as da
import scipy
import glob
import ufl
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator 
from . import arguments
from .pdeco import RVE
# from .constituitive import *


class Wave(RVE):
    def __init__(self, domain, case, mode, problem):
        super(Wave, self).__init__(domain, case, mode, problem)


    def opt_prepare(self):
        self.x_initial = np.array([0., 0., 0.5])
        self.bounds = np.array([[-0.2, 0.], [-0.1, 0.1], [0.4, 0.6]])
        self.maxiter = 3


    def compute_objective(self):
        lmd = 0.0
        H = fe.as_matrix([[-lmd, 0.], [0., -lmd]])
        self.RVE_solve(H)

        self.raw_num_eigs = 32
        self.actual_num_eigs = self.raw_num_eigs // 2
        # self.cut_len = self.actual_num_eigs // 2
        self.cut_len = 12

        self.all_eigen_vals, all_eigen_vecs, all_dlambdas = self.kV_path('irreducible')
        band_gap_grads = self.ks(self.all_eigen_vals)

        self.obj_val_AD = 0
        for i in range(self.all_eigen_vals.shape[0]):
            for j in range(self.all_eigen_vals.shape[1]):
                self.obj_val_AD += float(band_gap_grads[i, j]) * all_dlambdas[i][j]
 
        self.true_obj_val = self.compute_true_obj_val(self.all_eigen_vals)
        print(f'true_obj_val (true band gap) = {self.true_obj_val}')


    def kV_path(self, path):
        all_eigen_vals = []
        all_eigen_vecs = []
        all_dlambdas = []

        ks = []
        A = 2 * self.L0 
        if path == 'irreducible':
            num_k = 31
            kts = np.linspace(0, 3, num_k)
            for kt in kts:
                if kt < 1:
                    kx = (np.pi / A) * kt
                    ky = 0.
                elif kt < 2:
                    kx = np.pi / A
                    ky = np.pi / A * (kt - 1)
                else:
                    kx = np.pi / A * (3 - kt)
                    ky = kx
                ks.append((kx, ky))
        else:
            raise ValueError(f'Unknown path: {path}')

        for kx, ky in ks:
            kV = da.Constant((kx, ky))
            eigen_vals, eigen_vecs = self.eigen_solver(kV)
            all_eigen_vals.append(eigen_vals)
            all_eigen_vecs.append(eigen_vecs)
            dlambdas = []
            for ne in range(self.actual_num_eigs):
                dlambda = self.lamda_derivative(eigen_vals[ne], eigen_vecs[ne], kV)
                dlambdas.append(dlambda)
            all_dlambdas.append(dlambdas)

        return np.array(all_eigen_vals), np.array(all_eigen_vecs), all_dlambdas


    def lamda_derivative(self, eigen_val, eigen_vec, kV):
        eigen_u = da.Function(self.PP)
        eigen_u.vector()[:] = eigen_vec
        uR, uI = eigen_u.split()
        uAu = da.assemble(self.assemble_A(uR, uI, uR, uI, kV))
        uMu = da.assemble(self.assemble_M(uR, uI, uR, uI))
        # Remark(Tianju): dolfin_dajoint seems to have a bug here
        # float() must be applied, otherwise it loses dependency
        dlambda = (uAu - float(eigen_val) * uMu) / float(uMu)
        return dlambda


    def compute_true_obj_val(self, band_vals):
        band_vals = np.absolute(band_vals)  
        eigen_vals_low_band = band_vals[:, :self.cut_len]
        eigen_vals_high_band = band_vals[:, self.cut_len:]
        return np.max(eigen_vals_low_band) - np.min(eigen_vals_high_band)


    def ks(self, band_vals):
        '''Kreisselmeier–Steinhauser approximation of min and max'''
        band_vals = jnp.array(band_vals)
        def band_gap(band_vals):
            eigen_vals_low_band = band_vals[:, :self.cut_len]
            eigen_vals_high_band = band_vals[:, self.cut_len:]
            rho = 1e3
            max_low_band = 1 / rho * jax.scipy.special.logsumexp(rho * eigen_vals_low_band)
            min_high_band = -1 / rho * jax.scipy.special.logsumexp(-rho * eigen_vals_high_band)
            return max_low_band - min_high_band

        self.obj_val = band_gap(band_vals)
        print(f"obj_val (approximated band gap) = {self.obj_val}")
        band_gap_grads = jax.grad(band_gap)(band_vals)

        return np.array(band_gap_grads)


    def assemble_A(self, uR, uI, vR, vI, kV):
        i, j, k, l = ufl.indices(4)
        a = self.L[i, j, k, l] * (fe.grad(uR)[k, l] * fe.grad(vR)[i, j] + fe.grad(uI)[k, l] * fe.grad(vI)[i, j] + \
                                  kV[l] * kV[j] * (uR[k] * vR[i] + uI[k] * vI[i]) + \
                                  (uR[k] * fe.grad(vI)[i, j] - uI[k] * fe.grad(vR)[i, j]) * kV[l] - \
                                  (vI[i] * fe.grad(uR)[k, l] - vR[i] * fe.grad(uI)[k, l]) * kV[j]) * fe.dx
        return a


    def assemble_M(self, uR, uI, vR, vI):
         m = self.rho * (fe.inner(uR, vR) * fe.dx + fe.inner(uI, vI) * fe.dx)
         return m


    def eigen_solver(self, kV):  
        P = fe.VectorElement('CG', self.mesh.ufl_cell(), 1)
        self.PP = fe.FunctionSpace(self.mesh, P * P, constrained_domain=self.exterior_periodic)

        uR, uI = fe.TrialFunctions(self.PP)
        vR, vI = fe.TestFunctions(self.PP)
  
        a = self.assemble_A(uR, uI, vR, vI, kV)
        m = self.assemble_M(uR, uI, vR, vI)

        bcs = [da.DirichletBC(self.PP.sub(0), da.Constant((0., 0.)), self.corner, method='pointwise'),
               da.DirichletBC(self.PP.sub(1), da.Constant((0., 0.)), self.corner, method='pointwise')]

        
        # Assemble stiffness form
        A = fe.PETScMatrix()
        M = fe.PETScMatrix()
        fe.assemble(a, tensor=A)
        fe.assemble(m, tensor=M)

        # Create eigensolver
        solver = fe.SLEPcEigenSolver(A, M)
        solver.parameters["solver"] = "krylov-schur"
        solver.parameters["problem_type"] = "gen_hermitian"
        solver.parameters["spectrum"] = "target magnitude"
        solver.parameters["spectral_transform"] = "shift-and-invert"
        solver.parameters["spectral_shift"] = -1e5

        solver.solve(self.raw_num_eigs)
        
        eigen_vals = []
        eigen_vecs = []
        for i in range(self.raw_num_eigs):
            r, c, rx, cx = solver.get_eigenpair(i)
            eigen_vals.append(r)
            eigen_vecs.append(rx)

        eigen_vals = np.array(eigen_vals)
        eigen_vecs = np.array(eigen_vecs)

        # inds = eigen_vals.argsort()
        # eigen_vals = eigen_vals[inds]
        # eigen_vecs = eigen_vecs[inds]
 
        if self.actual_num_eigs < self.raw_num_eigs:
            odd_even_diff = np.sum(np.absolute(eigen_vals[0::2] - eigen_vals[1::2]))
            assert  odd_even_diff < 1e-3, f'odd eigenvalues differ too much from even ones, distance: {odd_even_diff}'
            print(f"Passing even-odd test")
            eigen_vals = eigen_vals[0::2]
            eigen_vecs = eigen_vecs[0::2]

        return eigen_vals, eigen_vecs


    def forward_runs(self):
        x_files = glob.glob(f'data/numpy/{self.domain}/{self.case}/{self.mode}/x_*')
        x_opt = np.load(sorted(x_files)[-1])
        x_initial = np.array([0., 0., 0.5])

        self.build_mesh()
        self.move_mesh(x_opt)
        self.compute_objective()

        print(self.all_eigen_vals.T)
        omegas = np.sqrt(np.absolute(self.all_eigen_vals))

        plt.figure()
        for eigen_vals in self.all_eigen_vals.T:
            plt.scatter(np.arange(len(eigen_vals)), eigen_vals, s=20., color='black')

        plt.figure()
        for omega in omegas.T:
            plt.scatter(np.arange(len(omega)), omega, s=20., color='black')

        plt.show()
   

    def debug(self):
        h = np.array([0, 1e-5, 0])
        x = np.array([-0.1, 0.1, 0.5])

        self.build_mesh()
        self.move_mesh(x)
        self.compute_objective()
        f = self.obj_val
        control = da.Control(self.delta)
        dJdm = da.compute_gradient(self.obj_val_AD, control).values()

        x_h = x + h
        self.build_mesh()
        self.move_mesh(x_h)
        self.compute_objective()
        f_h = self.obj_val

        x_2h = x + 2*h
        self.build_mesh()
        self.move_mesh(x_2h)
        self.compute_objective()
        f_2h = self.obj_val

        print(f"\n")
        print(f"f = {f}, dJdm = {dJdm}")
        print(f"f_h - f = {f_h - f}, f_2h - f = {f_2h - f}")
        print(f"r_h = {f_h - f - np.dot(dJdm, h)}, r_2h = {f_2h - f - np.dot(dJdm, 2*h)}")
        print(f"finite difference: {(f_h - f)/h[1]}")


def main():
    pde = Wave(domain='rve', case='wave', mode='band', problem='forward')    
    pde.run()


if __name__ == '__main__':
    main()
    plt.show()