import fenics as fe
import dolfin_adjoint as da
import glob
import numpy as np
import matplotlib.pyplot as plt
from . import arguments
from .pdeco import RVE
from .constituitive import *


class Auxetic(RVE):
    def __init__(self, domain, case, mode, problem):
        super(Auxetic, self).__init__(domain, case, mode, problem)


    def opt_prepare(self):
        self.x_ini = np.array([0., 0., 0.5])
        self.bounds = np.array([[-0.2, 0.], [-0.1, 0.1], [0.5, 0.5]])
        self.maxstep = 100


    def compute_objective(self):
        mapped_J = mapped_J_wrapper(self.s)
        if self.mode == 'normal':
            H = fe.as_matrix([[-0.04, 0.], [0., -0.1]])
            self.RVE_solve(H)
            PK_11 = da.assemble(self.PK_stress[0, 0] * mapped_J * fe.dx)
            self.J = PK_11**2
        elif self.mode == 'shear':
            H = fe.as_matrix([[0., 0.3], [0., -0.125]])
            self.RVE_solve(H)
            PK_12 = da.assemble(self.PK_stress[0, 1] * mapped_J * fe.dx)
            self.J = PK_12**2
        elif self.mode == 'min_energy':
            H = fe.as_matrix([[0., 0.], [0., -0.1]])
            self.RVE_solve(H)
            energy = da.assemble(self.E)
            self.J = energy
            print(f"energy = {energy}")
        elif self.mode == 'max_energy':
            H = fe.as_matrix([[0., 0.], [0., -0.1]])
            self.RVE_solve(H)
            force = da.assemble(self.PK_stress[1, 1] * mapped_J * fe.dx)
            energy = -1e2*da.assemble(self.E)
            self.J = energy
            print(f"energy = {energy}")
        elif self.mode == 'von-mises':
            self.RVE_solve(fe.as_matrix([[0., 0.], [0., -0.1]]))
            force = da.assemble(self.PK_stress[1, 1] * mapped_J * fe.dx)
            von_mises = 1e-6*da.assemble(self.sigma_v**4 * mapped_J * fe.dx)
            self.J = von_mises + 1e-1 * (force + 2.995)**2
            self.vm_s = da.project(self.sigma_v, self.S)
            self.vm_s.rename("s", "s")
            print(f"von_mises = {von_mises}, force = {force}")
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

        self.obj_val = self.J
        self.obj_val_AD = self.J


    def forward_runs_plot_energy(self, boundary_disp, energy, after_opt):
        fig = plt.figure()
        plt.plot(boundary_disp, energy/(self.L0*2)**2/self.young_modulus, marker='o', color='black')
        plt.tick_params(labelsize=14)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if self.mode == 'normal':
            xlabel = r'$\overline{H}_{11}$'
        else:
            xlabel = r'$\overline{H}_{12}$'

        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(r'$\overline{W}/E$', fontsize=16)
        # plt.legend()
        # plt.yticks(rotation=90)
        fig.savefig(f'data/pdf/{self.domain}/{self.case}/{self.mode}_energy_after_opt_{after_opt}.pdf', bbox_inches='tight')


    def forward_runs(self):
        variable_values = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/var_vals.npy')
        x_opt = variable_values[-1]
        after_opt = False
        if self.mode == 'normal':
            x_rep = np.array([0., 0., 0.5])
        elif self.mode == 'shear':
            x_rep = np.array([-0.2, 0.1, 0.5])
        else:
            raise ValueError(f'Unknown mode: {self.mode}')   
        x = x_opt if after_opt else x_rep

        self.build_mesh()
        self.move_mesh(x)
        vtkfile_mesh = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/u.pvd')

        cache = False
        if cache:
            boundary_disp = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/disp_after_opt_{after_opt}.npy')
            energy = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/energy_after_opt_{after_opt}.npy')
            force = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/force_after_opt_{after_opt}.npy')
        else:
            energy = []
            force = []
            if self.mode == 'normal':
                # TODO(Tianju): -0.08 is twice -0.04, make that a variable rather than a magic number!
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
                boundary_disp = np.linspace(-0.5, 0.5, 21)
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

            np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/disp_after_opt_{after_opt}.npy', boundary_disp)

            if len(energy) > 0:
                energy = np.array(energy)
                np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/energy_after_opt_{after_opt}.npy', energy)

            if len(force) > 0:
                force = np.array(force)
                np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/force_after_opt_{after_opt}.npy', force)
 
            print(f'energy = {energy}')
            print(f'force = {force}')

        self.forward_runs_plot_energy(boundary_disp, energy, after_opt)


    def debug(self):
        pass


def main():
    # pde = Auxetic(domain='rve', case='auxetic', mode='shear', problem='post-processing')    
    # pde.run()
    pde = Auxetic(domain='rve', case='auxetic', mode='normal', problem='forward')    
    pde.run()


if __name__ == '__main__':
    main()
    # plt.show()
