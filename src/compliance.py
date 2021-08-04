import fenics as fe
import dolfin_adjoint as da
import mshr
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from pyadjoint.overloaded_type import create_overloaded_object
from . import arguments
from .dr_homo import DynamicRelaxSolve
from .pdeco import PDECO
from .constituitive import *


class Compliance(PDECO):
    def __init__(self, domain, case, mode, problem):
        self.domain = domain
        self.case = case
        self.mode = mode
        self.young_modulus = 100
        self.poisson_ratio = 0.3
        super(Compliance, self).__init__(problem)

 
    def move_mesh(self, h_values=None):
        b_mesh = da.BoundaryMesh(self.mesh, "exterior")
        self.S_b = fe.VectorFunctionSpace(b_mesh, 'P', 1)
        self.h = da.Function(self.S_b, name="h")

        if h_values is not None:
            if hasattr(h_values, "__len__"):
                if len(h_values) == 1:
                    h_values = h_values[0]

            self.h.vector()[:] = h_values

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

        S = fe.VectorFunctionSpace(self.mesh, 'P', 1)

        u, v = fe.TrialFunction(S), fe.TestFunction(S)

        def epsilon(u):
            return fe.sym(fe.grad(u))

        def sigma(u, mu=1., lmb=0.):
            return 2 * mu * epsilon(u) + lmb * fe.tr(epsilon(u)) * fe.Identity(2)

        a = fe.inner(sigma(u, mu=mu), fe.grad(v)) * fe.dx
        L = fe.inner(h_V, v) * fe.ds
        bcs = [da.DirichletBC(S, da.Constant((0., 0.)), self.exterior)]   
        s = da.Function(S, name="mesh deformation")
        da.solve(a == L, s, bcs=bcs)

        return s


    def build_mesh(self, create_mesh=False): 
        L0 = 1.
        n_width = 8
        n_height = 2
        mesh_file = f'data/xdmf/{self.domain}/mesh/mesh.xdmf'
        if create_mesh:
            radius = 0.25 * L0
            resolution = 150
            material_domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(n_width * L0, n_height * L0))

            for i in range(n_width):
                for j in range(n_height):
                    material_domain -= mshr.Circle(fe.Point(L0 / 2. + i * L0, L0 / 2. + j * L0), radius)
                    if i > 0 and j > 0:
                        material_domain -= mshr.Circle(fe.Point(i * L0, j * L0), radius)

            self.mesh = mshr.generate_mesh(material_domain, resolution)
            with fe.XDMFFile(mesh_file) as file:
                file.write(self.mesh)
        else:
            self.mesh = fe.Mesh()
            with fe.XDMFFile(mesh_file) as file:
                file.read(self.mesh)

        # Add dolfin-adjoint dependency
        self.mesh = create_overloaded_object(self.mesh)

        class Exterior(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fe.near(x[0], L0 * n_width) or
                    fe.near(x[1], L0 * n_height) or
                    fe.near(x[0], 0) or
                    fe.near(x[1], 0))

        class Interior(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and x[0] > 0 and x[0] < L0 * n_width and x[1] > 0 and x[1] < L0 * n_height

        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], L0 * n_width)

        class Bottom(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Top(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], L0 * n_height)

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
        self.ds = fe.Measure('ds')(subdomain_data=boundaries)
        self.Vol0 = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))


    def forward_solve(self):
        V = fe.VectorFunctionSpace(self.mesh, 'P', 1)
        traction = da.Constant((0., -0.2))

        self.u = da.Function(V, name="v")
        du = fe.TrialFunction(V)
        v = fe.TestFunction(V)
        energy_density, PK_stress, _, _ = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, False)
        self.E = energy_density * fe.dx - fe.dot(traction, self.u) * self.ds(2)

        bcs = [da.DirichletBC(V, da.Constant((0., 0.)), self.left)]

        dE = fe.derivative(self.E, self.u, v)
        jacE = fe.derivative(dE, self.u, du)
 
        da.solve(dE == 0, self.u, bcs, J=jacE)
        self.disp = self.u

        if self.problem == 'forward':
            vtkfile_mesh = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/u.pvd')
            vtkfile_mesh << self.disp

        comp = da.assemble(fe.dot(traction, self.u) * self.ds(2))

        alpha = 1.
        Vol = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        reg = alpha * (Vol - self.Vol0)**2
        self.J = comp + reg

        print(f"obj val = {float(self.J)}, compliance = {float(comp)}, reg = {float(reg)}")

        return float(self.J)


    def forward_runs(self):
        self.build_mesh(True)
        self.forward_solve()


    def adjoint_optimization(self):
        print(f"\n###################################################################")
        print(f"Optimizing {self.case} - {self.mode}")
        print(f"###################################################################")

        self.object_values = []

        vtkfile_mesh = fe.File(f'data/pvd/{self.domain}/{self.case}/{self.mode}/{self.problem}/u.pvd')

        def objective(x):
            print(f"h abs max = {np.max(np.absolute(x))}")
            self.build_mesh()
            self.move_mesh(x)
            obj_val = self.forward_solve()
            vtkfile_mesh << self.disp
            objective.count += 1
            self.object_values.append(obj_val)
            return obj_val

        objective.count = 0

        def derivative(x):
            control = da.Control(self.h)
            dJdm = da.compute_gradient(self.J, control)
            da.set_working_tape(da.Tape())
            return dJdm.vector()[:]

        x_initial = 0.
        options = {'maxiter': 10, 'disp': True}   
        res = opt.minimize(fun=objective,
                           x0=x_initial,
                           method='CG',
                           jac=derivative,
                           callback=None,
                           options=options)

        np.save(f'data/numpy/{self.domain}/{self.case}/{self.mode}/obj_vals.npy', np.array(self.object_values))
 

def main():
    pde = Compliance(domain='beam', case='compliance', mode='simple', problem='forward')
    pde.run()    
    pde = Compliance(domain='beam', case='compliance', mode='simple', problem='inverse')
    pde.run()
    pde = Compliance(domain='beam', case='compliance', mode='simple', problem='post-processing')
    pde.run()


if __name__ == '__main__':
    main()
    plt.show()
