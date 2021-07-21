import fenics as fe
import dolfin_adjoint as da
import mshr
import numpy as np
import matplotlib.pyplot as plt
from pyadjoint.overloaded_type import create_overloaded_object
from . import arguments
from .dr_homo import DynamicRelaxSolve
from .pdeco import PDECO
from .constituitive import *


class Auxetic(PDECO):
    def __init__(self, problem, mode):
        self.case_name = "auxetic"
        self.young_modulus = 100
        self.poisson_ratio = 0.3
        self.mode = mode
        super(Auxetic, self).__init__(problem)


    def build_mesh(self): 
        mesh_file = 'data/xdmf/RVE_mesh/RVE.xdmf'
        self.mesh = fe.Mesh()
        with fe.XDMFFile(mesh_file) as file:
            file.read( self.mesh)

        # Add dolfin-adjoint dependency
        self.mesh = create_overloaded_object(self.mesh)

        self.L0 = 0.5
        L0 = self.L0
        n_cells = 2

        class Exterior(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fe.near(x[0], L0 * n_cells) or
                    fe.near(x[1], L0 * n_cells) or
                    fe.near(x[0], 0) or
                    fe.near(x[1], 0))

        class Interior(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and x[0] > 0 and x[0] < L0 * n_cells and x[1] > 0 and x[1] < L0 * n_cells

        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], L0 * n_cells)

        class Bottom(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Top(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], L0 * n_cells)

        class LowerLeft(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], 0) and fe.near(x[1], 0)

        class ExteriorPeriodic(fe.SubDomain):
            def __init__(self):
                super(ExteriorPeriodic, self).__init__(map_tol=1e-5)

            def inside(self, x, on_boundary):
                is_left = fe.near(x[0], 0)
                is_bottom = fe.near(x[1], 0)
                return is_left or is_bottom

            def map(self, x, y):
                if fe.near(x[0], L0 * n_cells):
                    y[0] = x[0] - L0 * n_cells
                    y[1] = x[1]
                elif fe.near(x[1], L0 * n_cells):
                    y[0] = x[0] 
                    y[1] = x[1] - L0 * n_cells
                else:
                    y[0] = 1000 #???
                    y[1] = 1000


        class InteriorPeriodic(fe.SubDomain):
            def __init__(self):
                super(InteriorPeriodic, self).__init__(map_tol=1e-5)

            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 0 and x[0] < L0 and x[1] > 0 and x[1] < L0

            def map(self, x, y):
                if x[0] > L0 and x[1] > L0:
                    # y[0] = x[0] - L0
                    # y[1] = x[1] - L0
                    y[0] = x[0] - x[0] // L0 * L0
                    y[1] = x[1] - x[1] // L0 * L0
                else:
                    y[0] = 1000
                    y[1] = 1000


        # class InteriorPeriodic(fe.SubDomain):
        #     def __init__(self):
        #         super(InteriorPeriodic, self).__init__(map_tol=1e-5)

        #     def inside(self, x, on_boundary):
        #         first_pore = on_boundary and x[0] > 0 and x[0] < L0 and x[1] > 0 and x[1] < L0
        #         ref_arc = x[0] >= L0/2. and x[1] >= L0/2. 
        #         return first_pore and ref_arc

        #     def map(self, x, y):
        #         if x[0] > 0 and x[0] < L0 * n_cells and x[1] > 0 and x[1] < L0 * n_cells:
        #             x0 = x[0] - x[0] // L0 * L0
        #             x1 = x[1] - x[1] // L0 * L0
        #             if x0 >= L0/2. and x1 >= L0/2.:
        #                 y[0] = x0
        #                 y[1] = x1                   
        #             elif x0 <= L0/2. and x1 >= L0/2.:
        #                 y[0] = L0 - x0
        #                 y[1] = x1
        #             elif x0 <= L0/2. and x1 <= L0/2.:
        #                 y[0] = L0 - x0
        #                 y[1] = L0 - x1
        #             else:
        #                 y[0] = x0
        #                 y[1] = L0 - x1                
        #         else:
        #             y[0] = 1000 #???
        #             y[1] = 1000


        self.interior_periodic = InteriorPeriodic()
        self.exterior_periodic = ExteriorPeriodic()
        self.corner = LowerLeft()
        self.exterior = Exterior()
        self.interior = Interior()
        self.left = Left()
        self.right = Right()
        self.bottom = Bottom()
        self.top = Top()

        boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        boundaries.set_all(0)
        self.left.mark(boundaries, 1)
        self.right.mark(boundaries, 2)
        self.bottom.mark(boundaries, 3)
        self.top.mark(boundaries, 4)
        # self.interior.mark(boundaries, 3)
        self.ds = fe.Measure('ds')(subdomain_data=boundaries)
        self.normal = fe.FacetNormal(self.mesh)
        self.Vol0 = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))


    def RVE_solve(self, H):

        V = fe.VectorFunctionSpace(self.mesh, 'P', 1, constrained_domain=self.exterior_periodic)
        V_non_periodic = fe.VectorFunctionSpace(self.mesh, 'P', 1)
        self.S = fe.FunctionSpace(self.mesh, 'DG', 0)

        self.u = da.Function(V, name="v")
        du = fe.TrialFunction(V)
        v = fe.TestFunction(V)

        energy_density, self.PK_stress, self.sigma_v = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, True, True, H)
        self.E = energy_density * fe.dx
        bcs = [da.DirichletBC(V, da.Constant((0., 0.)), self.corner, method='pointwise')]
        dE = fe.derivative(self.E, self.u, v)
        jacE = fe.derivative(dE, self.u, du)

        nIters, convergence = DynamicRelaxSolve(dE, self.u, bcs, jacE)
        da.solve(dE == 0, self.u, bcs, J=jacE)

        X = fe.SpatialCoordinate(self.mesh)
        self.disp = da.project(self.u + fe.dot(H, X), V_non_periodic)
        self.disp.rename("u", "u")

        if self.problem == 'forward':
            xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case_name}/{self.problem}/sols.xdmf')    
            xdmf_file_sols.write(self.u)

        if self.problem == 'debug':
            e = da.assemble(self.E)
            f = da.assemble(self.PK_stress[1, 1]*self.ds(4))
            print(f"Energy = {e}")
            print(f"force = {f}")
            self.energy.append(e)
            self.force.append(f)
            # self.xdmf_file_sols.write(self.disp, self.step)


    def forward_solve(self):
        alpha = 1e3
        Vol = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        reg = alpha * (Vol - self.Vol0)**2

        if self.mode == 'normal':
            H = fe.as_matrix([[-0.1, 0.], [0., -0.125]])
            self.RVE_solve(H)
            self.J = da.assemble(self.PK_stress[0, 0]*self.ds(1))**2 + da.assemble(self.PK_stress[0, 0]*self.ds(2))**2
        elif self.mode == 'shear':
            H = fe.as_matrix([[0., 0.36], [0., -0.125]])
            self.RVE_solve(H)
            self.J = da.assemble(self.PK_stress[0, 1]*self.ds(3))**2 + da.assemble(self.PK_stress[0, 1]*self.ds(4))**2
        elif self.mode == 'buckle':
            # h = 1e-2
            # base = -0.04
            # H_left = fe.as_matrix([[0., 0.], [0., base + h]])
            # H_center = fe.as_matrix([[0., 0.], [0., base]])
            # H_right = fe.as_matrix([[0., 0.], [0., base - h]])
            # self.RVE_solve(H_center)
            # f_center = da.assemble(self.PK_stress[1, 1]*self.ds(4))
            # self.RVE_solve(H_left)
            # f_left = da.assemble(self.PK_stress[1, 1]*self.ds(4))
            # self.RVE_solve(H_right)
            # f_right = da.assemble(self.PK_stress[1, 1]*self.ds(4))
            # self.J = -(f_right - 2*f_center + f_left)**2
            # print(f"f_left = {f_left}, f_center={f_center}, f_right = {f_right}")
            # print(f"left_diff = {f_center - f_left}, right_diff = {f_right - f_center}")

            H = fe.as_matrix([[0., 0.], [0., -0.1]])
            self.RVE_solve(H)
            force = da.assemble(self.PK_stress[1, 1]*fe.dx)
            energy = da.assemble(self.E)
            self.J = energy + reg
            print(f"force = {force}")
            print(f"Vol = {Vol}")

            ## -0.3504, -2.728
            # error = 0
            # boundary_disp = np.array([-0.01, -0.1])
            # # target_f = np.array([-0.3, -2.5])
            # target_f = np.array([-0.4, -3.4])

            # # boundary_disp = np.array([-0.01])
            # # target_f = np.array([-0.4])
            # error = []

            # for H22, tf in zip(boundary_disp, target_f):
            #     H = fe.as_matrix([[0., 0.], [0., H22]])
            #     self.RVE_solve(H)
            #     force = da.assemble(self.PK_stress[1, 1]*fe.dx)
            #     print(f"H22 = {H22}, f = {force}")
            #     error.append(((force - tf) / tf)**2)

            # self.J = error[0] + error[1] + reg
            # print(f"Vol = {Vol}")
            # print(f"error = {error}")

        elif self.mode == 'von-mises':
            self.RVE_solve(fe.as_matrix([[0., 0.], [0., -0.125]]))
            force =  1e3*da.assemble(self.PK_stress[1, 1]*self.ds(4))
            von_mises = 1e-1*da.assemble(self.sigma_v**4 * fe.dx)
            self.J = von_mises + force + reg
            self.s = da.project(self.sigma_v, self.S)
            self.s.rename("s", "s")
            print(f"von_mises={von_mises}, force={force}, reg={reg}")
        else:
            raise ValueError('Unknown mode!')

        print(f"obj val = {float(self.J)}\n")

        return float(self.J)


    def force_disp(self, h=None):
        self.energy = []
        self.force = []
        self.build_mesh()
        self.move_mesh(h)
        boundary_disp = np.linspace(0, -0.1, 11)
        for H22 in boundary_disp:
            H = fe.as_matrix([[0., 0.], [0., H22]])
            self.RVE_solve(H)
            # self.step += 1

        return self.energy, self.force


    def debug(self):
        # self.xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case_name}/{self.problem}/sols.xdmf')
        # self.step = 0

        h = np.load(f'data/numpy/{self.case_name}/h_49.npy')
        base_energy, base_force = self.force_disp()
        opt_energy, opt_force = self.force_disp(h)

        fig = plt.figure(0)
        plt.plot(base_energy, linestyle='--', marker='o', color='blue')
        plt.plot(opt_energy, linestyle='--', marker='o', color='red')        
        fig = plt.figure(1)
        plt.plot(base_force, linestyle='--', marker='o', color='blue')
        plt.plot(opt_force, linestyle='--', marker='o', color='red')
        plt.show()


def main():
    # pde = Auxetic(problem='debug', mode='normal')
    pde = Auxetic(problem='inverse', mode='buckle')    
    pde.run()


if __name__ == '__main__':
    main()
