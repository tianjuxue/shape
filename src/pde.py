import fenics as fe
import dolfin_adjoint as da
import math
import numpy as np
import shutil
import matplotlib.pyplot as plt
import mshr
from pyadjoint.overloaded_type import create_overloaded_object
from .dr_homo import DynamicRelaxSolve
from . import arguments


class PDECO(object):
    def __init__(self, problem):  
        self.problem = problem 
        self.periodic = None
        self.preparation()
        

    def run(self, opt_step=0):
        if self.problem == 'inverse':
            # self.build_mesh()
            # self.move_mesh()
            # self.RVE_solve()
            self.adjoint_optimization()
        elif self.problem == 'forward':
            self.opt_step = opt_step
            self.build_mesh()
            self.RVE_solve()
        elif self.problem == 'post-processing':
            self.opt_step = opt_step
            self.plot_force_displacement()
            plt.ioff()
            plt.show()
        elif self.problem == 'debug':
            self.opt_step = opt_step
            self.build_mesh()
            self.RVE_solve()
        else:
            raise ValueError('Unknown problem mode!')


    def preparation(self):
        if self.problem == 'inverse':
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.case_name}/{self.problem}', ignore_errors=True)
            shutil.rmtree(f'data/xdmf/{self.case_name}/{self.problem}', ignore_errors=True)


    def move_mesh(self, h_values=None):
        b_mesh = da.BoundaryMesh(self.mesh, "exterior")
        self.S_b = fe.VectorFunctionSpace(b_mesh, 'P', 1)
        self.h = da.Function(self.S_b, name="h")

        if h_values is not None:
            self.h.vector()[:] = h_values

        # self.h.vector()[:] = 0.5

        s = self.mesh_deformation(self.h)
        fe.ALE.move(self.mesh, s)

        return s


    def mesh_deformation(self, h):
        h_V = da.transfer_from_boundary(h, self.mesh)
        h_V.rename("Volume extension of h", "")

        V = fe.FunctionSpace(self.mesh, 'P', 1)
        u, v = fe.TrialFunction(V), fe.TestFunction(V)
        a = -fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        l = da.Constant(0.) * v * fe.dx
        mu_min = da.Constant(1., name="mu_min")
        mu_max = da.Constant(2., name="mu_max")
        bcs = [da.DirichletBC(V, mu_min, self.exterior), da.DirichletBC(V, mu_max, self.interior)] 
        mu = da.Function(V, name="mesh deformation mu")
        da.solve(a == l, mu, bcs=bcs)

        # S = fe.VectorFunctionSpace(self.mesh, 'P', 1, constrained_domain=self.interior_periodic)
        S = fe.VectorFunctionSpace(self.mesh, 'P', 1)

        u, v = fe.TrialFunction(S), fe.TestFunction(S)

        def epsilon(u):
            return fe.sym(fe.grad(u))

        def sigma(u, mu=1., lmb=0.):
            return 2 * mu * epsilon(u) + lmb * fe.tr(epsilon(u)) * fe.Identity(2)

        a = fe.inner(sigma(u, mu=mu), fe.grad(v)) * fe.dx
        L = fe.inner(h_V, v) * self.ds
        bcs = [da.DirichletBC(S, da.Constant((0., 0.)), self.exterior)]   
        s = da.Function(S, name="mesh deformation")
        da.solve(a == L, s, bcs=bcs)

        return s

    def adjoint_optimization(self):
        self.object_values = []
        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.problem}/u.pvd')
        h_val = 0.
        initial_guess = 0.
        for step in range(300):
            self.build_mesh()
            s = self.move_mesh(h_val)
            obj_val = self.RVE_solve(initial_guess)
            vtkfile_mesh << self.disp
            control = da.Control(self.h)
            dJdm = da.compute_gradient(self.J, control, options={"riesz_representation": "L2"})
            h_val = h_val - 5*1e-3 * dJdm.vector()[:]
            initial_guess = self.u.vector()[:]
            print(f"current objective value={obj_val} at step {step}\n")
            self.object_values.append(obj_val)
            if obj_val < 1e-3:
                break

        self.plot_optimization_progress()


    def plot_optimization_progress(self):
        fig = plt.figure()
        plt.plot(self.object_values, linestyle='--', marker='o')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (Gradient descent steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        fig.savefig(f'data/pdf/{self.case_name}_obj.pdf', bbox_inches='tight')
        plt.show()



class Auxetic(PDECO):
    def __init__(self, problem):
        self.case_name = "auxetic"
        self.young_modulus = 100
        self.poisson_ratio = 0.3
        # self.H = fe.as_matrix([[-0.05, 0.], [0., -0.125]])
        self.H = fe.as_matrix([[-0.1, 0.], [0., -0.125]])

        self.L0 = 0.5
        self.n_cells = 2
        super(Auxetic, self).__init__(problem)


    def build_mesh(self): 
        mesh_file = 'data/xdmf/RVE_mesh/RVE.xdmf'
        self.mesh = fe.Mesh()
        with fe.XDMFFile(mesh_file) as file:
            file.read( self.mesh)

        # Add dolfin-adjoint dependency
        self.mesh = create_overloaded_object(self.mesh)

        # Defensive copy
        self.mesh_initial = fe.Mesh(self.mesh)

        L0 = self.L0 
        n_cells = self.n_cells

        class Exterior(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fe.near(x[1], L0 * n_cells) or
                    fe.near(x[0], L0 * n_cells) or
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
                    y[0] = x[0] - L0
                    y[1] = x[1] - L0
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
        # self.interior.mark(boundaries, 3)
        self.ds = fe.Measure('ds')(subdomain_data=boundaries)
        self.normal = fe.FacetNormal(self.mesh)


    def RVE_solve(self, initial_guess=None):
        V = fe.VectorFunctionSpace(self.mesh, 'P', 1, constrained_domain=self.exterior_periodic)
        V_non_periodic = fe.VectorFunctionSpace(self.mesh, 'P', 1)

        self.u = da.Function(V, name="v")
        du = fe.TrialFunction(V)
        v = fe.TestFunction(V)
        energy_density = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, False, self.H)
        E = energy_density * fe.dx
        bcs = [da.DirichletBC(V, da.Constant((0., 0.)), self.corner, method='pointwise')]
        dE = fe.derivative(E, self.u, v)
        jacE = fe.derivative(dE, self.u, du)

        if initial_guess is not None:
            self.u.vector()[:] = initial_guess

        nIters, convergence = DynamicRelaxSolve(dE, self.u, bcs, jacE)
        da.solve(dE == 0, self.u, bcs, J=jacE)

        if self.problem == 'forward':
            xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case_name}/{self.problem}/sols.xdmf')    
            xdmf_file_sols.write(self.u)

        _, PK_stress = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, True, self.H)

        self.J = da.assemble(PK_stress[0, 0]*self.ds(1))**2 + da.assemble(PK_stress[0, 0]*self.ds(2))**2
 
        X = fe.SpatialCoordinate(self.mesh)
        self.disp = fe.project(self.u + fe.dot(self.H, X), V_non_periodic)
        self.disp.rename("u", "u")

        return float(self.J)


def DeformationGradientFluctuation(v, H):
    grad_u = fe.grad(v) + H  
    I = fe.Identity(v.geometric_dimension())
    return I + grad_u


def RightCauchyGreen(F):
    return F.T * F


def NeoHookeanEnergyFluctuation(v, young_modulus, poisson_ratio, return_stress, H_list):
    shear_mod = young_modulus / (2 * (1 + poisson_ratio))
    bulk_mod = young_modulus / (3 * (1 - 2*poisson_ratio))
    d = v.geometric_dimension()
    F = DeformationGradientFluctuation(v, H_list)
    F = fe.variable(F)
    J = fe.det(F)
    Jinv = J**(-2 / 3)
    I1 = fe.tr(RightCauchyGreen(F))

    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2) 
 
    if return_stress:
        first_pk_stress = fe.diff(energy, F)
        return energy, first_pk_stress

    return energy


def main():
    pde = Auxetic('inverse')
    pde.run()


if __name__ == '__main__':
    main()