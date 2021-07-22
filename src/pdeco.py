import fenics as fe
import dolfin_adjoint as da
import numpy as np
import shutil
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import ufl
import glob
from . import arguments


# class PoreExpression(da.UserExpression):
#     def __init__(self, L0, delta):
#         super(PoreExpression, self).__init__()
#         self.L0 = L0
#         self.delta = delta

#     def eval(self, values, x):
#         L0 = self.L0
#         x0 = x[0] - x[0] // L0 * L0 - L0/2
#         x1 = x[1] - x[1] // L0 * L0 - L0/2
#         values[0] = float(self.delta)*x0
#         values[1] = float(self.delta)*x1

#     def value_shape(self):
#         return (2,)

# exp = PoreExpression(self.L0, delta)



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
            plt.ioff()
            plt.show()
        elif self.problem == 'debug':
            self.debug()
        else:
            raise ValueError(f'Unknown problem: {self.problem}')


    def preparation(self):
        if self.problem == 'inverse':
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}', ignore_errors=True)
            shutil.rmtree(f'data/xdmf/{self.case_name}/{self.mode}/{self.problem}', ignore_errors=True)
            numpy_files = glob.glob(f'data/numpy/{self.case_name}/{self.mode}/*')
            for f in numpy_files:
                os.remove(f)
 

    def compute_disp(self, params):
        x1, x2 = fe.SpatialCoordinate(self.mesh)
        x1 = fe.conditional(fe.gt(x1, self.L0), x1 - self.L0, x1)
        x2 = fe.conditional(fe.gt(x2, self.L0), x2 - self.L0, x2)
        x1 -= self.L0/2
        x2 -= self.L0/2

        r_raw =  fe.sqrt(x1**2 + x2**2)

        phi_0 = 0.5
        theta = ufl.atan_2(x2, x1)
        r0 = self.L0 * np.sqrt(2 * phi_0) / fe.sqrt(np.pi * (2 + params[0]**2 + params[1]**2))

        r = r0 * (1 + params[0] * fe.cos(4 * theta) + params[1] * fe.cos(8 * theta))
        delta_x1 = (r - r_raw) * fe.cos(theta)
        delta_x2 = (r - r_raw) * fe.sin(theta)

        return fe.as_vector([delta_x1, delta_x2])


    def move_mesh(self, delta_val):
        self.delta = da.Constant(delta_val)
        s = self.mesh_deformation(self.delta)
        fe.ALE.move(self.mesh, s)


    def mesh_deformation(self, delta):
        V = fe.FunctionSpace(self.mesh, 'P', 1)
        u, v = fe.TrialFunction(V), fe.TestFunction(V)
        a = -fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        l = da.Constant(0.) * v * fe.dx
        mu_min = da.Constant(1, name="mu_min")
        mu_max = da.Constant(2*1e2, name="mu_max")
        bcs = [da.DirichletBC(V, mu_min, self.exterior), da.DirichletBC(V, mu_max, self.interior)] 
        mu = da.Function(V, name="mesh deformation mu")
        da.solve(a == l, mu, bcs=bcs)

        S = fe.VectorFunctionSpace(self.mesh, 'P', 1)
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


    # def move_mesh(self, h_values=None):
    #     b_mesh = da.BoundaryMesh(self.mesh, "exterior")
    #     self.S_b = fe.VectorFunctionSpace(b_mesh, 'P', 1)
    #     self.h = da.Function(self.S_b, name="h")

    #     # self.h_size = len(self.h.vector()[:])

    #     if h_values is not None:
    #         if hasattr(h_values, "__len__"):
    #             if len(h_values) == 1:
    #                 h_values = h_values[0]

    #         self.h.vector()[:] = h_values

    #     # self.h.vector()[:] = 0.5

    #     s = self.mesh_deformation(self.h)
    #     fe.ALE.move(self.mesh, s)


    # def mesh_deformation(self, h):
    #     h_V = da.transfer_from_boundary(h, self.mesh)
    #     h_V.rename("Volume extension of h", "")

    #     V = fe.FunctionSpace(self.mesh, 'P', 1)
    #     u, v = fe.TrialFunction(V), fe.TestFunction(V)
    #     a = -fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    #     l = da.Constant(0.) * v * fe.dx
    #     mu_min = da.Constant(1., name="mu_min")
    #     mu_max = da.Constant(2., name="mu_max")
    #     bcs = [da.DirichletBC(V, mu_min, self.exterior), da.DirichletBC(V, mu_max, self.interior)] 
    #     mu = da.Function(V, name="mesh deformation mu")
    #     da.solve(a == l, mu, bcs=bcs)

    #     S = fe.VectorFunctionSpace(self.mesh, 'P', 1)

    #     u, v = fe.TrialFunction(S), fe.TestFunction(S)

    #     def epsilon(u):
    #         return fe.sym(fe.grad(u))

    #     def sigma(u, mu=1., lmb=0.):
    #         return 2 * mu * epsilon(u) + lmb * fe.tr(epsilon(u)) * fe.Identity(2)

    #     a = fe.inner(sigma(u, mu=mu), fe.grad(v)) * fe.dx
    #     L = fe.inner(h_V, v) * self.ds(5)
    #     bcs = [da.DirichletBC(S, da.Constant((0., 0.)), self.exterior)]   
    #     s = da.Function(S, name="mesh deformation")
    #     da.solve(a == L, s, bcs=bcs)

    #     return s


    def adjoint_optimization(self):
        print(f"\n###################################################################")
        print(f"Optimizing {self.case_name} - {self.mode}")
        print(f"###################################################################")

        self.object_values = []

        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/u.pvd')
        if self.mode == 'von-mises':
            vtkfile_stress = fe.File(f'data/pvd/{self.case_name}/{self.mode}/{self.problem}/s.pvd')

        def objective(x):
            print(f"h abs max = {np.max(np.absolute(x))}")
            print(f"x = {x}")

            self.build_mesh()
            self.move_mesh(x)
            obj_val = self.compute_objective()
            vtkfile_mesh << self.disp
            if self.mode == 'von-mises':
                vtkfile_stress << self.s
                max_vm_stress = np.max(self.s.vector()[:])
                objective_aux.append(max_vm_stress)
                print(f"max_vm_stress = {max_vm_stress}")

            np.save(f'data/numpy/{self.case_name}/{self.mode}/h_{objective.count:03}.npy', x)
            objective.count += 1
            self.object_values.append(obj_val)
            return obj_val

        objective.count = 0
        objective_aux = []

        # def derivative(x):
        #     control = da.Control(self.h)
        #     dJdm = da.compute_gradient(self.J, control)
        #     da.set_working_tape(da.Tape())   
        #     return dJdm.vector()[:]

        def derivative(x):
            control = da.Control(self.delta)
            dJdm = da.compute_gradient(self.J, control)
            da.set_working_tape(da.Tape())

            print(f"dJdm = {dJdm.values()}")
            # exit()

            return dJdm.values()


        # TODO: Magic number
        # num_x = 912
        # x_initial = np.zeros(num_x)
        # x_initial = 0.
        # bounds = bound * np.vstack([-np.ones(num_x), np.ones(num_x)]).T

        num_x = 2
        x_initial = np.array([-0., -0.])

        if self.mode == 'normal':
            bound = 1.
            method = 'L-BFGS-B'
        elif self.mode == 'shear':
            bound = 0.5
            method = 'L-BFGS-B'
        elif self.mode == 'min_energy':
            bound = 0.5
            method = 'L-BFGS-B'
        elif self.mode == 'max_energy':
            bound = 0.5
            method = 'L-BFGS-B'
        elif self.mode == 'von-mises':
            bound = 0.2
            method = 'L-BFGS-B'
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

 
        method = 'L-BFGS-B'
        bounds = np.array([[-0.2, 0.], [-0.1, 0.1]])


        options = {'maxiter': 100, 'disp': True}  # CG or L-BFGS-B or Newton-CG
        res = opt.minimize(fun=objective,
                           x0=x_initial,
                           method=method,
                           jac=derivative,
                           bounds=bounds,
                           callback=None,
                           options=options)

        np.save(f'data/numpy/{self.case_name}/{self.mode}/obj_vals.npy', np.array(self.object_values))
        if self.mode == 'von-mises':
            np.save(f'data/numpy/{self.case_name}/{self.mode}/vm_stress.npy', np.array(objective_aux))


    def plot_optimization_progress(self):
        object_values = np.load(f'data/numpy/{self.case_name}/{self.mode}/obj_vals.npy')
        fig = plt.figure()
        plt.plot(object_values, linestyle='--', marker='o')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (Gradient descent steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        fig.savefig(f'data/pdf/{self.case_name}/{self.mode}_{self.case_name}_obj.pdf', bbox_inches='tight')
        plt.show()

