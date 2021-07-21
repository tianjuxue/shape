import fenics as fe
import dolfin_adjoint as da
import numpy as np
import shutil
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import glob
from . import arguments


class SinExpression(fe.UserExpression):
    def __init__(self, L0):
        super(SinExpression, self).__init__()
        self.L0 = L0

    def eval(self, values, x):
        L0 = self.L0
        x0 = x[0] - x[0] // L0 * L0 - L0/2
        x1 = x[1] - x[1] // L0 * L0 - L0/2
        values[0] = -1*x0
        values[1] = -1*x1

    def value_shape(self):
        return (2,)



class PDECO(object):
    def __init__(self, problem):  
        self.problem = problem 
        self.preparation()
        

    def run(self):
        if self.problem == 'inverse':
            self.adjoint_optimization()
        elif self.problem == 'forward':
            self.build_mesh()
            self.move_mesh()
            self.forward_solve()
        elif self.problem == 'post-processing':
            self.plot_force_displacement()
            plt.ioff()
            plt.show()
        elif self.problem == 'debug':
            self.debug()
        else:
            raise ValueError('Unknown problem!')


    def preparation(self):
        if self.problem == 'inverse':
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}', ignore_errors=True)
            shutil.rmtree(f'data/xdmf/{self.case_name}/{self.mode}/{self.problem}', ignore_errors=True)
            numpy_files = glob.glob(f'data/numpy/{self.mode}/{self.case_name}/*')
            for f in numpy_files:
                os.remove(f)
 

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
        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/u.pvd')
        if self.mode == 'von-mises':
            vtkfile_stress = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/s.pvd')

        def objective(x):
            print(f"h abs max = {np.max(np.absolute(x))}")

            self.build_mesh()
            self.move_mesh(x)
            obj_val = self.forward_solve()
            vtkfile_mesh << self.disp
            if self.mode == 'von-mises':
                vtkfile_stress << self.s
            if self.mode == 'buckle':
                np.save(f'data/numpy/{self.case_name}/{self.mode}/h_{objective.count}.npy', x)
            objective.count += 1

            return obj_val

        objective.count = 0

        def derivative(x):
            control = da.Control(self.h)
            dJdm = da.compute_gradient(self.J, control)
            da.set_working_tape(da.Tape())   
            return dJdm.vector()[:]


        num_x = 912
        # x_initial = 0.
        x_initial = np.zeros(num_x)
        self.build_mesh()
        b_mesh = da.BoundaryMesh(self.mesh, "exterior")
        self.S_b = fe.VectorFunctionSpace(b_mesh, 'P', 1)
        exp = SinExpression(self.L0)
        self.h = da.interpolate(exp, self.S_b)
        x_initial = np.array(self.h.vector()[:])
        x_initial = np.zeros(num_x)

        self.build_mesh()
        self.move_mesh(x_initial)
        obj_val = self.forward_solve()
        control = da.Control(self.h)

        # # dh = da.Function(self.S_b)
        # # dh.vector()[:] = 0.01
        # # Jhat = da.ReducedFunctional(self.J, control)
        # # conv_rate = da.taylor_test(Jhat, self.h, dh)

        # dJdm = da.compute_gradient(self.J, control)
        # vtkfile_dJdm = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/dJdm.pvd')
        # vtkfile_dJdm << dJdm
        # exit()


        # vtkfile_h = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/h.pvd')
        # vtkfile_h << self.h
 

        # x = x_initial
        # for gd_step in range(100):
        #     print(f"h abs max = {np.max(np.absolute(x))}")
        #     self.build_mesh()
        #     self.move_mesh(x)
        #     obj_val = self.forward_solve()
        #     vtkfile_mesh << self.disp
        #     control = da.Control(self.h)
        #     dJdm = da.compute_gradient(self.J, control)
        #     da.set_working_tape(da.Tape()) 
        #     dJdm = dJdm.vector()[:]
        #     x = x - dJdm

        # exit()

        bounds = 1.*np.vstack([-np.ones(num_x), np.ones(num_x)]).T
        options = {'maxiter': 10, 'disp': True}  # CG > L-BFGS-B > Newton-CG
        res = opt.minimize(fun=objective,
                           x0=x_initial,
                           method='L-BFGS-B',
                           jac=derivative,
                           bounds=bounds,
                           callback=None,
                           options=options)


  

    def plot_optimization_progress(self):
        fig = plt.figure()
        plt.plot(self.object_values, linestyle='--', marker='o')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (Gradient descent steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        fig.savefig(f'data/pdf/{self.mode}/{self.case_name}_obj.pdf', bbox_inches='tight')
        plt.show()

