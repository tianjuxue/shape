import fenics as fe
import dolfin_adjoint as da
import numpy as np
import shutil
import matplotlib.pyplot as plt
import scipy.optimize as opt
from . import arguments


class PDECO(object):
    def __init__(self, problem):  
        self.problem = problem 
        self.periodic = None
        self.preparation()
        

    def run(self, opt_step=0):
        if self.problem == 'inverse':
            self.adjoint_optimization()
        elif self.problem == 'forward':
            self.opt_step = opt_step
            self.build_mesh()
            self.move_mesh()
            self.forward_solve()
        elif self.problem == 'post-processing':
            self.opt_step = opt_step
            self.plot_force_displacement()
            plt.ioff()
            plt.show()
        elif self.problem == 'debug':
            self.opt_step = opt_step
            self.build_mesh()
            self.forward_solve()
        else:
            raise ValueError('Unknown problem!')


    def preparation(self):
        if self.problem == 'inverse':
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.case_name}/{self.problem}', ignore_errors=True)
            shutil.rmtree(f'data/xdmf/{self.case_name}/{self.problem}', ignore_errors=True)


    def move_mesh(self, h_values=None):
        b_mesh = da.BoundaryMesh(self.mesh, "exterior")
        self.S_b = fe.VectorFunctionSpace(b_mesh, 'P', 1)
        self.h = da.Function(self.S_b, name="h")

        # self.h_size = len(self.h.vector()[:])

        if h_values is not None:
            if hasattr(h_values, "__len__"):
                if len(h_values) == 1:
                    h_values = h_values[0]

            self.h.vector()[:] = h_values

        # self.h.vector()[:] = 0.5

        s = self.mesh_deformation(self.h)
        fe.ALE.move(self.mesh, s)


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
        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.problem}/u.pvd')

        def objective(x):
            self.build_mesh()
            self.move_mesh(x)
            obj_val = self.forward_solve()
            vtkfile_mesh << self.disp
            return obj_val

        def derivative(x):
            control = da.Control(self.h)
            dJdm = da.compute_gradient(self.J, control, options={"riesz_representation": "L2"})
            da.set_working_tape(da.Tape())   
            return dJdm.vector()[:]

        x_initial = 0.
        options = {'maxiter': 10, 'disp': True}  # CG > BFGS > Newton-CG
        res = opt.minimize(fun=objective,
                           x0=x_initial,
                           method='CG',
                           jac=derivative,
                           callback=None,
                           options=options)
 

    def plot_optimization_progress(self):
        fig = plt.figure()
        plt.plot(self.object_values, linestyle='--', marker='o')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (Gradient descent steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        fig.savefig(f'data/pdf/{self.case_name}_obj.pdf', bbox_inches='tight')
        plt.show()

