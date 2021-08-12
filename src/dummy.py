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
        self.x_ini = np.array([0., 0., 0.5])
        self.bounds = np.array([[-0.2, 0.], [-0.1, 0.1], [0.5, 0.5]])
        self.maxstep = 100


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
        x = np.array([-0.05, -0.05, 0.55])
        H = fe.as_matrix([[0, 0.1], [0., -0.1]])

        def compute_objective(x_h):
            self.build_mesh()
            self.move_mesh(x_h)
            self.RVE_solve(H)
            obj_val_AD = da.assemble(self.E)
            control = da.Control(self.delta)
            dJdm = da.compute_gradient(obj_val_AD, control)
            return float(obj_val_AD), dJdm.values()

        cache = True
        if cache:
            hs = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/taylor_h.npy')
            r_zero = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/taylor_r0.npy')
            r_first = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/taylor_r1.npy')
        else:
            hs = [1e-5*0.5**i for i in range(4)]
            f, df = compute_objective(x)
            r_zero = []
            r_first = []
            for i in range(len(x)):
                r_zero.append([])
                r_first.append([])
                for h in hs:
                    x_h = np.array(x)
                    x_h[i] += h
                    f_h, _ = compute_objective(x_h)
                    r0 = np.absolute(f_h - f)
                    r1 = np.absolute(f_h - f - h*df[i])
                    r_zero[i].append(r0)
                    r_first[i].append(r1)

            np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/taylor_h.npy', np.array(hs))
            np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/taylor_r0.npy', np.array(r_zero))
            np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/taylor_r1.npy', np.array(r_first))

        def plot_r(hs, rs, name):
            fig = plt.figure()
            markers = ['o', 's', '^']
            colors = ['red', 'blue', 'green']
            labels = [r'$\xi_1$', r'$\xi_2$', r'$\phi_0$']
            for i in range(len(rs)):
                plt.loglog(hs, rs[i], linestyle='-', marker=markers[i], color=colors[i], label=labels[i])
            plt.tick_params(labelsize=16)
            plt.xlabel(r'$h$', fontsize=18)
            plt.ylabel(r'$r_{\textrm{' + name + r'}}$', fontsize=18)
            plt.legend(fontsize=16, frameon=False)

            if name == 'first':
                p1 = [3.5*1e-6, 6*1e-11]
                p2 = [4.5*1e-6, p1[1]]
                p3 = [p2[0], np.exp(2 * np.log(p2[0]/p1[0]))*p1[1]]
                plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], color='black')  
                plt.text(3.75*1e-6, 8*1e-11, '2', fontsize=16)
            else:
                p1 = [3.5*1e-6, 3.2*1e-6]
                p2 = [4.5*1e-6, p1[1]]
                p3 = [p2[0], np.exp(np.log(p2[0]/p1[0]))*p1[1]]
                plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], color='black')  
                plt.text(3.75*1e-6, 3.7*1e-6, '1', fontsize=16)                

            fig.savefig(f'data/pdf/{self.domain}/{self.case}/{self.mode}_residual_{name}.pdf', bbox_inches='tight')

        plot_r(hs, r_zero,'zeroth')
        plot_r(hs, r_first,'first')


    def forward_runs(self):
        # self.show_deformed_rve()
        # self.show_configs()
        self.taylor_test()
 

def main():
    pde = Dummy(domain='rve', case='dummy', mode='bc', problem='forward')    
    pde.run()


if __name__ == '__main__':
    main()
    plt.show()
