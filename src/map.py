import fenics as fe
import dolfin_adjoint as da
import glob
import numpy as np
import matplotlib.pyplot as plt
from . import arguments
from .pdeco import RVE
from .constituitive import *


class Map(RVE):
    '''
    根据IJNME审稿人第一轮意见major revision想补点关于map是否one-to-one的实验，但是好像没用上
    '''
    def __init__(self, domain, case, mode, problem):
        super(Map, self).__init__(domain, case, mode, problem)


    def forward_runs(self):
        prefix = f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/'

        x = np.array([0., 0, 0.5])
        H = fe.as_matrix([[0., 0.], [0., 0.]])
        self.build_mesh()
        self.move_mesh(x)
        self.RVE_solve(H)
        vtkfile = fe.File(prefix + 'u_ref.pvd')
        vtkfile << self.disp

        # x = np.array([-0.0, 0.1, 0.5])
        x = np.array([-0.0, 0.0, 0.6])
        H = fe.as_matrix([[0., 0.], [0., 0.]])
        self.build_mesh()
        self.move_mesh(x)
        self.RVE_solve(H)
        vtkfile = fe.File(prefix + 'u_lag.pvd')
        vtkfile << self.disp

        # x = np.array([-0.2, 0.1, 0.5])
        # H = fe.as_matrix([[0., 0.6], [0., -0.2]])
        # self.build_mesh()
        # self.move_mesh(x)
        # self.RVE_solve(H)
        # vtkfile = fe.File(prefix + 'u_eul.pvd')
        # vtkfile << self.disp


def main():
    pde = Map(domain='rve', case='map', mode='test', problem='forward')    
    pde.run()


if __name__ == '__main__':
    main()
