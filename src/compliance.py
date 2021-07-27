import fenics as fe
import dolfin_adjoint as da
import mshr
from pyadjoint.overloaded_type import create_overloaded_object
from . import arguments
from .dr_homo import DynamicRelaxSolve
from .pdeco import PDECO
from .constituitive import *


class Compliance(PDECO):
    def __init__(self, problem, mode):
        self.case_name = "compliance"
        self.young_modulus = 100
        self.poisson_ratio = 0.3
        self.mode = mode
        super(Compliance, self).__init__(problem)


    def build_mesh(self, create_mesh=False): 
        L0 = 1.
        n_width = 8
        n_height = 2
        mesh_file = f'data/xdmf/{self.case_name}/mesh/mesh.xdmf'
        if create_mesh:
            # radius = L0 / 4.
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
                file.read( self.mesh)


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
        energy_density, PK_stress, _, _ = NeoHookeanEnergyFluctuation(self.u, self.young_modulus, self.poisson_ratio, True, False)
        self.E = energy_density * fe.dx - fe.dot(traction, self.u) * self.ds(2)

        bcs = [da.DirichletBC(V, da.Constant((0., 0.)), self.left)]

        dE = fe.derivative(self.E, self.u, v)
        jacE = fe.derivative(dE, self.u, du)
 
        da.solve(dE == 0, self.u, bcs, J=jacE)
        self.disp = self.u

        if self.problem == 'forward':
            xdmf_file_sols = fe.XDMFFile(f'data/xdmf/{self.case_name}/{self.problem}/sols.xdmf')    
            xdmf_file_sols.write(self.u)

        comp = da.assemble(fe.dot(traction, self.u) * self.ds(2))

        alpha = 1.
        Vol = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        reg = alpha * (Vol - self.Vol0)**2
        self.J = comp + reg

        print(f"obj val = {float(self.J)}, compliance = {float(comp)}, reg = {float(reg)}")

        return float(self.J)


def main():
    pde = Compliance(problem='inverse', mode='beam')
    pde.run()


if __name__ == '__main__':
    main()
