import fenics as fe
import dolfin_adjoint as da
import mshr
import glob
import numpy as np
import matplotlib.pyplot as plt
from pyadjoint.overloaded_type import create_overloaded_object
from . import arguments
from .dr_homo import DynamicRelaxSolve
from .pdeco import PDECO
from .constituitive import *


class RVE(PDECO):
    def __init__(self, problem, mode):
        self.case_name = "rve"
        self.young_modulus = 100
        self.poisson_ratio = 0.3
        self.mode = mode
        super(RVE, self).__init__(problem)


    def build_mesh(self): 
        mesh_file = f'data/xdmf/{self.case_name}/mesh/mesh.xdmf'
        self.mesh = fe.Mesh()


        print(mesh_file)

        with fe.XDMFFile(mesh_file) as file:
            file.read(self.mesh)

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
        self.interior.mark(boundaries, 5)
        self.ds = fe.Measure('ds')(subdomain_data=boundaries)
        self.n = fe.FacetNormal(self.mesh)
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

        # if self.problem == 'forward':
        #     xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case_name}/{self.problem}/sols.xdmf')    
        #     xdmf_file_sols.write(self.u)

        # if self.problem == 'debug':
        #     e = da.assemble(self.E)
        #     f = da.assemble(self.PK_stress[1, 1]*self.ds(4))
        #     print(f"Energy = {e}")
        #     print(f"force = {f}")
        #     self.energy.append(e)
        #     self.force.append(f)
            # self.xdmf_file_sols.write(self.disp, self.step)


    def compute_objective(self):
        alpha = 1e3
        Vol = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        reg = alpha * (Vol - self.Vol0)**2

        reg = 0

        if self.mode == 'normal':
            H = fe.as_matrix([[-0.1, 0.], [0., -0.125]])
            self.RVE_solve(H)
            PK_11 = da.assemble(self.PK_stress[0, 0]*fe.dx)
            self.J = PK_11**2 + reg
        elif self.mode == 'shear':
            H = fe.as_matrix([[0., 0.35], [0., -0.125]])
            self.RVE_solve(H)
            PK_12 = da.assemble(self.PK_stress[0, 1]*fe.dx)
            self.J = 1e1*PK_12**2 + reg
        elif self.mode == 'min_energy':
            H = fe.as_matrix([[0., 0.], [0., -0.125]])
            self.RVE_solve(H)
            energy = da.assemble(self.E)
            self.J = energy + reg
            print(f"energy = {energy}")
        elif self.mode == 'max_energy':
            H = fe.as_matrix([[0., 0.], [0., -0.01]])
            self.RVE_solve(H)
            force = da.assemble(self.PK_stress[1, 1]*fe.dx)
            energy = -1e2*da.assemble(self.E)
            self.J = energy + reg
            print(f"energy = {energy}")
        elif self.mode == 'von-mises':
            self.RVE_solve(fe.as_matrix([[0., 0.], [0., -0.125]]))
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


    # def force_disp(self, h=None):
    #     self.energy = []
    #     self.force = []
    #     self.build_mesh()
    #     self.move_mesh(h)
    #     boundary_disp = np.linspace(0, -0.1, 11)
    #     for H22 in boundary_disp:
    #         H = fe.as_matrix([[0., 0.], [0., H22]])
    #         self.RVE_solve(H)
    #         # self.step += 1

    #     return self.energy, self.force

    def forward_runs(self):
        h_files = glob.glob(f'data/numpy/{self.case_name}/{self.mode}/h_*')
        h = np.load(sorted(h_files)[-1])
        self.build_mesh()
        self.move_mesh(h)
        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/u.pvd')

        energy = []
        force = []
        if self.mode == 'normal':
            boundary_disp = np.linspace(0, -0.2, 11)
            for H11 in boundary_disp:
                H = fe.as_matrix([[H11, 0.], [0., -0.125]])
                self.RVE_solve(H)
                vtkfile_mesh << self.disp
                e = da.assemble(self.E)
                energy.append(e)
                print(f" H11 = {H11}")
                print(f"e = {e}")
        elif self.mode == 'shear':
            boundary_disp = np.linspace(0, 0.6, 11)
            for H12 in boundary_disp:
                H = fe.as_matrix([[0., H12], [0., -0.125]])
                self.RVE_solve(H)
                vtkfile_mesh << self.disp
                e = da.assemble(self.E)
                energy.append(e)
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
            np.save(f'data/numpy/{self.case_name}/{self.mode}/energy.npy', np.array(energy))

        if len(force) > 0:
            np.save(f'data/numpy/{self.case_name}/{self.mode}/force.npy', np.array(force))
 

        print(f'energy = {energy}')
        print(f'force = {force}')



    def plot_forward_runs(self):
        force = np.load(f'data/numpy/{self.case_name}/{self.mode}/force.npy')
        energy = np.load(f'data/numpy/{self.case_name}/{self.mode}/energy.npy')
 
        fig = plt.figure(0)
        plt.plot(base_energy, linestyle='--', marker='o', color='blue')
        plt.plot(opt_energy, linestyle='--', marker='o', color='red')        
        fig = plt.figure(1)
        plt.plot(base_force, linestyle='--', marker='o', color='blue')
        plt.plot(opt_force, linestyle='--', marker='o', color='red')
        plt.show()


    def visualize_results(self):
        pass


def main():
    # pde = RVE(problem='inverse', mode='normal')
    # pde.run()

    case_names = ['normal', 'shear', 'min_energy', 'max_energy', 'von-mises']
    # modes = ['normal']
    for mode in modes:
        pde = RVE(problem='forward', mode=mode)    
        pde.run()


if __name__ == '__main__':
    main()
