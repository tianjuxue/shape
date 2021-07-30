import fenics as fe
import dolfin_adjoint as da
import glob
import numpy as np
import matplotlib.pyplot as plt
from . import arguments
from .pdeco import rve
# from .constituitive import *


class Auxetic(RVE):
    def __init__(self, case, mode, problem):
        super(Auxetic, self).__init__(case, mode, problem)


    def compute_objective(self):
        alpha = 1e3
        Vol = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        reg = alpha * (Vol - self.Vol0)**2

        reg = 0

        if self.mode == 'normal':
            H = fe.as_matrix([[-0.04, 0.], [0., -0.1]])
            self.RVE_solve(H)
            PK_11 = da.assemble(self.PK_stress[0, 0]*fe.dx)
            self.J = PK_11**2 + reg
        elif self.mode == 'shear':
            H = fe.as_matrix([[0., 0.3], [0., -0.125]])
            # H = fe.as_matrix([[0., 0.], [0., -0.125]])
            self.RVE_solve(H)
            PK_12 = da.assemble(self.PK_stress[0, 1]*fe.dx)
            self.J = PK_12**2 + reg
        elif self.mode == 'min_energy':
            H = fe.as_matrix([[0., 0.], [0., -0.1]])
            self.RVE_solve(H)
            energy = da.assemble(self.E)
            self.J = energy + reg
            print(f"energy = {energy}")
        elif self.mode == 'max_energy':
            H = fe.as_matrix([[0., 0.], [0., -0.1]])
            self.RVE_solve(H)
            force = da.assemble(self.PK_stress[1, 1]*fe.dx)
            energy = -1e2*da.assemble(self.E)
            self.J = energy + reg
            print(f"energy = {energy}")
        elif self.mode == 'von-mises':
            self.RVE_solve(fe.as_matrix([[0., 0.], [0., -0.1]]))
            force = da.assemble(self.PK_stress[1, 1]*fe.dx)
            von_mises = 1e-6*da.assemble(self.sigma_v**4 * fe.dx)
            self.J = von_mises + 1e-1 * (force + 2.995)**2 + reg
            self.s = da.project(self.sigma_v, self.S)
            self.s.rename("s", "s")
            print(f"von_mises = {von_mises}, force = {force}")
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

        print(f"reg = {reg}, Vol = {Vol}")
        print(f"obj val = {float(self.J)}\n")

        return float(self.J)


    def forward_runs(self):
        h_files = glob.glob(f'data/numpy/{self.case}/{self.mode}/h_*')
        h = np.load(sorted(h_files)[-1])
        self.build_mesh()
        self.move_mesh(h)
        vtkfile_mesh = fe.File(f'data/pvd/{self.case}/{self.mode}/{self.problem}/u.pvd')

        energy = []
        force = []
        if self.mode == 'normal':
            boundary_disp = np.linspace(0, -0.08, 11)
            for H11 in boundary_disp:
                H = fe.as_matrix([[H11, 0.], [0., -0.1]])
                self.RVE_solve(H)
                vtkfile_mesh << self.disp
                e = da.assemble(self.E)
                energy.append(e)
                print(f"H11 = {H11}")
                print(f"e = {e}")
        elif self.mode == 'shear':
            boundary_disp = np.linspace(0, 0.6, 11)
            for H12 in boundary_disp:
                H = fe.as_matrix([[0., H12], [0., -0.125]])
                self.RVE_solve(H)
                vtkfile_mesh << self.disp
                e = da.assemble(self.E)
                energy.append(e)
                print(f"H12 = {H12}")
                print(f"e = {e}")
        elif self.mode == 'min_energy' or self.mode == 'max_energy':
            boundary_disp = np.linspace(0, -0.125, 11)
            for H22 in boundary_disp:
                H = fe.as_matrix([[0., 0.], [0., H22]])
                self.RVE_solve(H)
                vtkfile_mesh << self.disp
                f = da.assemble(self.PK_stress[1, 1]*self.ds(4))
                force.append(f)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

        if len(energy) > 0:
            np.save(f'data/numpy/{self.case}/{self.mode}/energy.npy', np.array(energy))

        if len(force) > 0:
            np.save(f'data/numpy/{self.case}/{self.mode}/force.npy', np.array(force))
 

        print(f'energy = {energy}')
        print(f'force = {force}')


    def plot_forward_runs(self):
        force = np.load(f'data/numpy/{self.case}/{self.mode}/force.npy')
        energy = np.load(f'data/numpy/{self.case}/{self.mode}/energy.npy')
 
        fig = plt.figure(0)
        plt.plot(base_energy, linestyle='--', marker='o', color='blue')
        plt.plot(opt_energy, linestyle='--', marker='o', color='red')        
        fig = plt.figure(1)
        plt.plot(base_force, linestyle='--', marker='o', color='blue')
        plt.plot(opt_force, linestyle='--', marker='o', color='red')
        plt.show()


    def visualize_results(self):
        pass


    def debug(self):
        self.build_mesh()
        vtkfile_F = fe.File(f'data/pvd/{self.case}/{self.mode}/{self.problem}/F.pvd')
        D = fe.TensorFunctionSpace(self.mesh, 'DG', 0)
        delta = da.Constant([0.1, 0.1])
        s = self.mesh_deformation(delta)
        F = fe.Identity(2) + fe.grad(s)
        proj_1 = da.project(F, D)
        fe.ALE.move(self.mesh, s)
        F_inv = fe.Identity(2) - fe.grad(s)
        proj_2 = da.project(F_inv * proj_1, D)
        proj_2.rename("F", "F")
        vtkfile_F << proj_2


def main():
    # pde = RVE(case='rve', problem='inverse', mode='normal')
    # pde.run()

    # modes = ['normal', 'shear', 'max_energy', 'min_energy', 'von-mises']
    modes = ['normal']
    for mode in modes:
        pde = RVE(domain='rve', case='auxetic', mode=mode, problem='debug')    
        pde.run()


if __name__ == '__main__':
    main()
