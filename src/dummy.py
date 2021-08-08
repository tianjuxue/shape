import fenics as fe
import dolfin_adjoint as da
import glob
import numpy as np
import matplotlib.pyplot as plt
from . import arguments
from .pdeco import RVE
from .constituitive import *


class Dummy(RVE):
    def __init__(self, domain, case, mode, problem):
        super(Dummy, self).__init__(domain, case, mode, problem)


    def opt_prepare(self):
        self.x_initial = np.array([0., 0., 0.5])
        self.bounds = np.array([[-0.2, 0.], [-0.1, 0.1], [0.5, 0.5]])
        self.maxiter = 100


    def show_deformed_rve(self):
        x = np.array([0., 0., 0.5])
        self.build_mesh()
        self.move_mesh(x)
        vtkfile_sols = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/deformed.pvd')
        H = fe.as_matrix([[0, 0.1], [0., -0.1]])
        self.RVE_solve(H)
        vtkfile_sols << self.disp


    def show_configs(self):
        vtkfile_mesh = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/configs.pvd')
        x = np.array([0., 0., 0.5])
        H = fe.as_matrix([[0, 0.], [0., 0.]])
        self.build_mesh()
        self.move_mesh(x)
        self.RVE_solve(H)
        vtkfile_mesh << self.disp
        x = np.array([-0.05, -0.05, 0.55])
        self.build_mesh()
        self.move_mesh(x)
        self.RVE_solve(H)
        vtkfile_mesh << self.disp
        H = fe.as_matrix([[0, 0.1], [0., -0.1]])
        self.build_mesh()
        self.move_mesh(x)
        self.RVE_solve(H)
        vtkfile_mesh << self.disp


    def taylor_test(self):
        
        

    def forward_runs(self):
        # self.show_deformed_rve()
        self.show_configs()
 

def main():
    pde = Dummy(domain='rve', case='dummy', mode='bc', problem='forward')    
    pde.run()


if __name__ == '__main__':
    main()

