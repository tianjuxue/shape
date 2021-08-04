import fenics as fe
import dolfin_adjoint as da
import numpy as np
import shutil
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import ufl
import glob
from pyadjoint.overloaded_type import create_overloaded_object
from .dr_homo import DynamicRelaxSolve
from . import arguments
from .constituitive import *


class PDECO(object):
    def __init__(self, problem):  
        self.problem = problem 
        self.preparation()
        

    def run(self):
        if self.problem == 'inverse':
            self.adjoint_optimization()
        elif self.problem == 'forward':
            self.forward_runs()
        elif self.problem == 'post-processing':
            self.visualize_results()
        elif self.problem == 'debug':
            self.debug()
        else:
            raise ValueError(f'Unknown problem: {self.problem}')


    def preparation(self):
        if self.problem == 'inverse':
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}', ignore_errors=True)
            numpy_files = glob.glob(f'data/numpy/{self.domain}/{self.case}/{self.mode}/*')
            for f in numpy_files:
                os.remove(f)
 

    def visualize_results(self):
        object_values = np.load(f'data/numpy/{self.domain}/{self.case}/{self.mode}/obj_vals.npy')
        fig = plt.figure()
        plt.plot(object_values, linestyle='--', marker='o', color='black')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (Optimization steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        fig.savefig(f'data/pdf/{self.domain}/{self.case}/{self.mode}_obj.pdf', bbox_inches='tight')


class RVE(PDECO):
    def __init__(self, domain, case, mode, problem):
        self.domain = domain
        self.case = case
        self.mode = mode
        self.young_modulus = 1e3
        self.poisson_ratio = 0.3
        self.rho = 1.
        super(RVE, self).__init__(problem)


    def build_mesh(self): 
        mesh_file = f'data/xdmf/{self.domain}/mesh/mesh.xdmf'
        self.mesh = fe.Mesh()

        print(f'Load mesh file {mesh_file}')

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


    def compute_disp(self, params):
        x1, x2 = fe.SpatialCoordinate(self.mesh)
        x1 = fe.conditional(fe.gt(x1, self.L0), x1 - self.L0, x1)
        x2 = fe.conditional(fe.gt(x2, self.L0), x2 - self.L0, x2)
        x1 -= self.L0/2
        x2 -= self.L0/2

        r_raw = fe.sqrt(x1**2 + x2**2)

        # phi_0 = 0.5
        # phi_0 = params[2]
        theta = ufl.atan_2(x2, x1)
        r0 = self.L0 * fe.sqrt(2 * params[2]) / fe.sqrt(np.pi * (2 + params[0]**2 + params[1]**2))
        r = r0 * (1 + params[0] * fe.cos(4 * theta) + params[1] * fe.cos(8 * theta))
        delta_x1 = (r - r_raw) * fe.cos(theta)
        delta_x2 = (r - r_raw) * fe.sin(theta)

        return fe.as_vector([delta_x1, delta_x2])


    def move_mesh(self, delta_val):
        self.delta = da.Constant(delta_val)
        s = self.mesh_deformation(self.delta)
        fe.ALE.move(self.mesh, s)


    def mesh_deformation(self, delta):
        V = fe.FunctionSpace(self.mesh, 'CG', 1)
        u, v = fe.TrialFunction(V), fe.TestFunction(V)
        a = -fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        l = da.Constant(0.) * v * fe.dx
        mu_min = da.Constant(1, name="mu_min")
        mu_max = da.Constant(2*1e2, name="mu_max")
        bcs = [da.DirichletBC(V, mu_min, self.exterior), da.DirichletBC(V, mu_max, self.interior)] 
        mu = da.Function(V, name="mesh deformation mu")
        da.solve(a == l, mu, bcs=bcs)

        S = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        u, v = fe.TrialFunction(S), fe.TestFunction(S)
        s = da.Function(S, name="mesh deformation")

        def epsilon(u):
            return fe.sym(fe.grad(u))

        def sigma(u, mu, lmb):
            return 2 * mu * epsilon(u) + lmb * fe.tr(epsilon(u)) * fe.Identity(2)

        stress = lambda u: sigma(u, mu, mu)

        def penalty(u):
            F = fe.Identity(u.geometric_dimension()) + fe.grad(u)
            F = fe.variable(F)
            J = fe.det(F)
            energy = 1e2*(J - 1)**4
            first_pk_stress = fe.diff(energy, F)
            return first_pk_stress

        exp = self.compute_disp(delta)
        eta = 1e5
        a = fe.inner(stress(s), fe.grad(v)) * fe.dx - fe.dot(fe.dot(stress(s), self.n), v) * self.ds(5) - \
            fe.dot(fe.dot(stress(v), self.n), s) * self.ds(5) + eta * fe.dot(s, v) * self.ds(5) + \
            fe.dot(fe.dot(stress(v), self.n), exp) * self.ds(5) - eta * fe.dot(exp, v) * self.ds(5) + \
            fe.inner(penalty(s), fe.grad(v)) * fe.dx
    
        bcs = [da.DirichletBC(S, da.Constant((0., 0.)), self.exterior)]   
        da.solve(a == 0, s, bcs=bcs)

        return s


    def adjoint_optimization(self):
        print(f"\n###################################################################")
        print(f"Optimizing {self.domain} - {self.case} - {self.mode}")
        print(f"###################################################################")

        self.object_values = []

        vtkfile_mesh = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/u.pvd')
        if self.mode == 'von-mises':
            vtkfile_stress = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/s.pvd')

        def objective(x):
            print(f"x = {x}")

            self.build_mesh()
            self.move_mesh(x)
            self.compute_objective()
            print(f"J = {self.obj_val}")
 
            vtkfile_mesh << self.disp
            if self.mode == 'von-mises':
                vtkfile_stress << self.s
                max_vm_stress = np.max(self.s.vector()[:])
                objective_aux.append(max_vm_stress)
                print(f"max_vm_stress = {max_vm_stress}")

            np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/x_{objective.count:03}.npy', x)
            objective.count += 1
            self.object_values.append(self.obj_val)
            return self.obj_val

        objective.count = 0
        objective_aux = []

        def derivative(x):
            control = da.Control(self.delta)
            dJdm = da.compute_gradient(self.obj_val_AD, control)
            da.set_working_tape(da.Tape())

            print(f"dJdm = {dJdm.values()}")

            return dJdm.values()

        # self.x_initial = np.array([-0.2, 0.1, 0.5])
        # bounds = np.array([[-0.2, 0.], [-0.1, 0.1], [0.45, 0.55]])
        # self.bounds = np.array([[-0.2, -0.2], [0.1, 0.1], [0.4, 0.6]])
        # bounds = np.array([[0., 0.], [0., 0.], [0.5, 0.5]])

        self.opt_prepare()

        options = {'maxiter': self.maxiter, 'disp': True}  # CG or L-BFGS-B or Newton-CG
        res = opt.minimize(fun=objective,
                           x0=self.x_initial,
                           method='L-BFGS-B',
                           jac=derivative,
                           bounds=self.bounds,
                           callback=None,
                           options=options)

        np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/obj_vals.npy', np.array(self.object_values))
        if self.mode == 'von-mises':
            np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/vm_stress.npy', np.array(objective_aux))


    def RVE_solve(self, H, solve=True):
        self.V = fe.VectorFunctionSpace(self.mesh, 'CG', 1, constrained_domain=self.exterior_periodic)
        V_non_periodic = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.S = fe.FunctionSpace(self.mesh, 'DG', 0)

        self.u = da.Function(self.V, name="v")
        self.du = fe.TrialFunction(self.V)
        self.v = fe.TestFunction(self.V)

        energy_density, self.PK_stress, self.L, self.sigma_v = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, True, H)
        self.E = energy_density * fe.dx
        self.bcs = [da.DirichletBC(self.V, da.Constant((0., 0.)), self.corner, method='pointwise')]
        dE = fe.derivative(self.E, self.u, self.v)
        self.jacE = fe.derivative(dE, self.u, self.du)

        if solve:
            nIters, convergence = DynamicRelaxSolve(dE, self.u, self.bcs, self.jacE)
            da.solve(dE == 0, self.u, self.bcs, J=self.jacE)

        X = fe.SpatialCoordinate(self.mesh)
        self.disp = da.project(self.u + fe.dot(H, X), V_non_periodic)
        self.disp.rename("u", "u")

        # if self.problem == 'forward':
        #     xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case}/{self.problem}/sols.xdmf')    
        #     xdmf_file_sols.write(self.u)
