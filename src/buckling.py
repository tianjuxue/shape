import fenics as fe
import dolfin_adjoint as da
import scipy
import glob
import numpy as np
import matplotlib.pyplot as plt
from . import arguments
from .pdeco import RVE
from .constituitive import *


class Buckling(RVE):
    def __init__(self, domain, case, mode, problem):
        super(Buckling, self).__init__(domain, case, mode, problem)


    def opt_prepare(self, x=None):
        if self.mode == 'poreA':
            self.x_initial = np.array([0., 0., 0.5])
            self.bounds = np.array([[0., 0.], [0., 0.], [0.4, 0.6]])
            self.selected_eigen_number = 2
            self.critical_lmd = 0.02
            self.indices = [2, 3, 5]
            self.ylim = (-6000, 11000)
        elif self.mode == 'poreB':
            self.x_initial = np.array([-0.2, 0.1, 0.5])
            self.bounds = np.array([[-0.2, -0.2], [0.1, 0.1], [0.4, 0.6]])  
            self.selected_eigen_number = 2 
            self.critical_lmd = 0.03 
            self.indices = [2, 4, 5]  
            self.ylim = (-3000, 8000)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')            

        self.maxiter = 100
        print(f"Opt prepare...")
        self.build_mesh()
        if x is None:
            self.move_mesh(self.x_initial)
        else:
            self.move_mesh(x)
        self.compute_objective_helper(lmd=0., initial=True)


    def compute_objective(self):
        self.compute_objective_helper(lmd=self.critical_lmd, initial=False)


    def compute_objective_helper(self, lmd, initial):
        self.H = fe.as_matrix([[-lmd, 0.], [0., -lmd]])
        self.RVE_solve(self.H, solve=False)
        dummy_l = fe.dot(da.Constant((0., 0.)), self.v) * fe.dx(domain=self.mesh)
        A = fe.PETScMatrix()
        M = fe.PETScMatrix()
        m = self.rho * fe.inner(self.du, self.v) * fe.dx
        fe.assemble(m, tensor=M)
        fe.assemble(self.jacE, tensor=A)   
        # b = fe.PETScVector()
        # fe.assemble_system(self.jacE, dummy_l, self.bcs, A_tensor=A, b_tensor=b)

        solver = fe.SLEPcEigenSolver(A, M)
        solver.parameters["solver"] = "krylov-schur"
        solver.parameters["spectrum"] = "target magnitude"
        solver.parameters["spectral_transform"] = "shift-and-invert"
        solver.parameters["spectral_shift"] = -1e6

        num_solves = 32
        num_candidates = 6
        solver.solve(num_solves)
        
        self.eigen_vals = []
        self.eigen_vecs = []
        for i in range(num_candidates):
            r, c, rx, cx = solver.get_eigenpair(i)
            self.eigen_vals.append(r)
            self.eigen_vecs.append(rx)

        print(self.eigen_vals)

        if initial:
            self.initial_eigen_vecs = self.eigen_vecs
            if self.problem == 'inverse':
                vtkfile_eigen = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/eigen.pvd')
                eigen_u = fe.Function(self.V, name='e')
                for eigen_vec in self.eigen_vecs:
                    eigen_u.vector()[:] = eigen_vec
                    # eigen_u.rename("e", "e")
                    vtkfile_eigen << eigen_u
        else:
            self.eigen_vals, self.eigen_vecs = self.sort_by_initial_eigenvecs(self.eigen_vals, self.eigen_vecs)

        self.selected_eigen_vec = self.eigen_vecs[self.selected_eigen_number]
        self.selected_eigen_val = self.eigen_vals[self.selected_eigen_number]

        print(f"self.selected_eigen_val = {self.selected_eigen_val}")
   
        dlambda = self.lamda_derivative(self.selected_eigen_val, self.selected_eigen_vec)
        self.obj_val_AD = 2 * float(self.selected_eigen_val) * dlambda
        self.obj_val = float(self.selected_eigen_val)**2


    def sort_by_initial_eigenvecs(self, eigen_vals, eigen_vecs):
        sorted_eigen_vals = []
        sorted_eigen_vecs = []

        for num in range(len(eigen_vals)):
            max_proj = np.absolute((np.dot(eigen_vecs[0].get_local(), self.initial_eigen_vecs[num].get_local())))
            matching_index = 0
            for i in range(len(eigen_vecs)):
                tmp_proj = np.absolute((np.dot(eigen_vecs[i].get_local(), self.initial_eigen_vecs[num].get_local())))
                if tmp_proj > max_proj:
                    matching_index = i
                    max_proj = tmp_proj

            sorted_eigen_vals.append(eigen_vals[matching_index])
            sorted_eigen_vecs.append(eigen_vecs[matching_index])

        return sorted_eigen_vals, sorted_eigen_vecs

 
    def lamda_derivative(self, eigen_val, eigen_vec):
        eigen_u = da.Function(self.V)
        eigen_u.vector()[:] = eigen_vec
        dE = fe.derivative(self.E, self.u, eigen_u)
        jacE = fe.derivative(dE, self.u, eigen_u)
        uAu = da.assemble(jacE)
        # uu = np.dot(eigen_vec.get_local(), eigen_vec.get_local())
        # dlambda = uAu / float(uu)
        uMu = da.assemble(fe.inner(eigen_u, eigen_u) * fe.dx)
        dlambda = (uAu - float(eigen_val) * uMu) / float(uMu)
        return dlambda
 

    def forward_runs_helper(self):
        selected_eigen_vals = []
        all_eigen_vals = []
        self.lmds = np.linspace(0, 0.05, 21)
        for lmd in self.lmds:
            self.compute_objective_helper(lmd, False)
            all_eigen_vals.append(self.eigen_vals)
            selected_eigen_vals.append(self.selected_eigen_val)
        all_eigen_vals = np.array(all_eigen_vals)
        return all_eigen_vals, selected_eigen_vals


    def forward_runs_plot_eigenvalues(self, all_eigen_vals, name):
        markers = ['o', 's', '^']
        colors = ['red', 'blue', 'green']
        fig = plt.figure()
        for i in range(len(self.indices)):     
            plt.ylim(self.ylim)
            plt.plot(self.lmds, all_eigen_vals.T[self.indices[i]], marker=markers[i], color=colors[i])
        plt.plot(self.lmds, np.zeros_like(self.lmds), color='black')
        plt.tick_params(labelsize=14)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("$\lambda$ (compression)", fontsize=16)
        plt.ylabel("$\omega^2$ (eigenvalue)", fontsize=16)
        # plt.legend()
        # plt.yticks(rotation=90)
        fig.savefig(f'data/pdf/{self.domain}/{self.case}/{self.mode}_{name}_eigens.pdf', bbox_inches='tight')
        

    def forward_runs(self):
        self.opt_prepare()
        initial_all_eigen_vals, initial_selected_eigen_vals = self.forward_runs_helper()

        x_files = glob.glob(f'data/numpy/{self.domain}/{self.case}/{self.mode}/x_*')
        x_opt = np.load(sorted(x_files)[-1])
        self.opt_prepare(x_opt)
        opt_all_eigen_vals, opt_selected_eigen_vals = self.forward_runs_helper()

        self.forward_runs_plot_eigenvalues(initial_all_eigen_vals, 'ini')
        self.forward_runs_plot_eigenvalues(opt_all_eigen_vals, 'opt')

        self.visualize_results()
        
        plt.show()


    def debug(self):
        h = np.array([0, 0, 1e-5])
        x = np.array([0, 0, 0.45])

        self.selected_eigen_number = 0
        self.build_mesh()
        self.move_mesh(x)
        self.compute_objective_helper(lmd=.02, initial=True)
        f = self.obj_val

        control = da.Control(self.delta)
        dJdm = da.compute_gradient(self.obj_val_AD, control).values()

        x_h = x + h
        self.build_mesh()
        self.move_mesh(x_h)
        self.compute_objective_helper(lmd=.02, initial=True)
        f_h = self.obj_val

        x_2h = x + 2*h
        self.build_mesh()
        self.move_mesh(x_2h)
        self.compute_objective_helper(lmd=.02, initial=True)
        f_2h = self.obj_val

        print(f"\n")
        print(f"f = {f}, dJdm = {dJdm}")
        print(f"f_h - f = {f_h - f}, f_2h - f = {f_2h - f}")
        print(f"r_h = {f_h - f - np.dot(dJdm, h)}, r_2h = {f_2h - f - np.dot(dJdm, 2*h)}")
        print(f"finite difference: {(f_h - f)/h[-1]}")



def main():
    pde = Buckling(domain='rve', case='buckling', mode='poreB', problem='forward')    
    pde.run()


if __name__ == '__main__':
    main()
