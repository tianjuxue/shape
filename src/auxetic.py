import fenics as fe
import dolfin_adjoint as da
import mshr
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
        if self.mode == 'normal':
            self.H = fe.as_matrix([[-0.1, 0.], [0., -0.125]])
        elif self.mode == 'shear':
            self.H = fe.as_matrix([[0., 0.36], [0., -0.125]])
        else:
            raise ValueError('Unknown mode!')

        super(Auxetic, self).__init__(problem)


    def build_mesh(self): 
        mesh_file = 'data/xdmf/RVE_mesh/RVE.xdmf'
        self.mesh = fe.Mesh()
        with fe.XDMFFile(mesh_file) as file:
            file.read( self.mesh)

        # Add dolfin-adjoint dependency
        self.mesh = create_overloaded_object(self.mesh)

        L0 = 0.5
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


    def forward_solve(self):
        V = fe.VectorFunctionSpace(self.mesh, 'P', 1, constrained_domain=self.exterior_periodic)
        V_non_periodic = fe.VectorFunctionSpace(self.mesh, 'P', 1)

        self.u = da.Function(V, name="v")
        du = fe.TrialFunction(V)
        v = fe.TestFunction(V)
        energy_density, PK_stress = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, True, True, self.H)
        self.E = energy_density * fe.dx
        bcs = [da.DirichletBC(V, da.Constant((0., 0.)), self.corner, method='pointwise')]
        dE = fe.derivative(self.E, self.u, v)
        jacE = fe.derivative(dE, self.u, du)
 
        nIters, convergence = DynamicRelaxSolve(dE, self.u, bcs, jacE)
        da.solve(dE == 0, self.u, bcs, J=jacE)

        X = fe.SpatialCoordinate(self.mesh)
        self.disp = da.project(self.u + fe.dot(self.H, X), V_non_periodic)
        self.disp.rename("u", "u")

        if self.problem == 'forward':
            xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case_name}/{self.problem}/sols.xdmf')    
            xdmf_file_sols.write(self.disp)

        if self.mode == 'normal':
            self.J = da.assemble(PK_stress[0, 0]*self.ds(1))**2 + da.assemble(PK_stress[0, 0]*self.ds(2))**2
        elif self.mode == 'shear':
            self.J = da.assemble(PK_stress[0, 1]*self.ds(3))**2 + da.assemble(PK_stress[0, 1]*self.ds(4))**2
        else:
            raise ValueError('Unknown mode!')

        print(f"obj val = {float(self.J)}")
        return float(self.J)


def main():
    pde = Auxetic(problem='forward', mode='shear')
    pde.run()


if __name__ == '__main__':
    main()
