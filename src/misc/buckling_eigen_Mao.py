"""
FEniCS tutorial demo program: Linear elastic problem.

  -div(sigma(u)) = f

The model is used to simulate an elastic beam clamped at
its left end and deformed under its own weight.
"""

from __future__ import print_function
from fenics import *
from geometry import convert_geometry
from fem import Geometric_Parameters
from buckling import BucklingProblem, ElasticProblem
from buckling import KfKs, cal_func_KfKs, cal_Lame
from buckling import cal_psi_NeoHookean

##########################################################################

def OptSolver():
    # Create the PETScTAOSolver
    solver = PETScTAOSolver()
    # Set some parameters
    solver.parameters["method"] = "tron"
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True
    solver.parameters["maximum_iterations"] = 100

    #solver.parameters["linear_solver"] = "mumps"
    # Uncomment this line to see the available parameters
    #info(parameters, True)
    # Parse (PETSc) parameters
    #parameters.parse()
    return solver

def ElasticSolver():
    # create Newton solver
    solver =  NewtonSolver()
    solver.parameters["linear_solver"] = "mumps"
    #solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6
    solver.parameters["absolute_tolerance"] = 1e-10
    solver.parameters["maximum_iterations"] = 20
    return solver 

def EigenSolver(elasticProblem):
    A = PETScMatrix()
    assemble(elasticProblem.a, tensor=A)
    for bc in elasticProblem.bcs:
        bc.apply(A)
    # calculating the eigen values
    # Create eigensolver
    solver = SLEPcEigenSolver(A)
    solver.parameters["solver"] = "krylov-schur"
    solver.parameters["tolerance"] = 1e-9
    solver.parameters["problem_type"] = "hermitian"
    solver.parameters["spectrum"] = "target real"
    #solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"
    solver.parameters["spectral_shift"] = -5.0
    return solver


# eigen solver to determine the wavelength
def eigen_solve(solver):
    # Assemble stiffness form
    neigs = 10
    solver.solve(neigs)
    r, _ , u, _ = solver.get_eigenpair(0)
    return r, u



# eigen solver to determine the wavelength
def eigen_solver(J, bcs):
    # Assemble stiffness form
    A = PETScMatrix()
    dummy = v[0]*dx
    assemble_system(J, dummy, bcs, A_tensor=A)
    
    # calculating the eigen values
    # Create eigensolver
    solver = SLEPcEigenSolver(A)
    solver.parameters["solver"] = "krylov-schur"
    solver.parameters["tolerance"] = 1e-9
    solver.parameters["problem_type"] = "hermitian"
    solver.parameters["spectrum"] = "target real"
    #solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"
    solver.parameters["spectral_shift"] = -5.0
    neigs = 10
    solver.solve(neigs)
    
    #for i1 in range(min(neigs, solver.get_number_converged())):
    #    r, _ = solver.get_eigenvalue(i1) # ignore the imaginary part
    #    print(r)
    r, _ , u, _ = solver.get_eigenpair(0)
    return r, u

##########################################################################

# Geometric parameters
L = 60.0; W = 2.0; Hf = 1.0; Hs = 9.0; H = Hf + Hs;
# Physical parameters
Ef = 5.0; nuF = 0.30; Es=1.0; nuS = 0.30;

# Create mesh and define function space
#mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 20, 6, 6)
fileDir = "./"
GeoParams = Geometric_Parameters(L = L,
                                 W = W,
                                 Hf = Hf,
                                 Hs = Hs,
                                 dHf = Hf,
                                 dHs = H/2.0)

#if MPI.rank(mpi_comm_world()) == 0:
convert_geometry(fileDir, GeoParams)

mesh = Mesh("geometry.xml")
markers = MeshFunction("size_t", mesh, fileDir + "geometry_physical_region.xml")
V = VectorFunctionSpace(mesh, 'P', 2)
VS = FunctionSpace(mesh, 'P', 2)

print('total number of DOF: %d' % ( len(V.dofmap().dofs()) ) )

##########################################################################

# Define boundary condition

def leftBoundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def rightBoundary(x, on_boundary):
    return on_boundary and near(x[0], L, 1e-6)

def frontBoundary(x, on_boundary):
    return on_boundary and near(x[1], 0.0)

def backBoundary(x, on_boundary):
    return on_boundary and near(x[1], W)

def bottomBoundary(x, on_boundary):
    return on_boundary and near(x[2], 0.0)

def fixedBoundary1(x, on_boundary):
    return near(x[0], 0.0) and near(x[2], 0.0)

def fixedBoundary2(x, on_boundary):
    return near(x[0], L) and near(x[2], 0.0)

bcLeft = DirichletBC(V.sub(0), Constant(0), leftBoundary)
bcRight = DirichletBC(V.sub(0), Constant(0), rightBoundary)
bcFront = DirichletBC(V.sub(1), Constant(0), frontBoundary)
bcBack = DirichletBC(V.sub(1), Constant(0), backBoundary)
bcBottom = DirichletBC(V.sub(2), Constant(0), bottomBoundary)
#bcFixed1 = DirichletBC(V.sub(2), Constant(0), fixedBoundary1, method='pointwise')
#bcFixed2 = DirichletBC(V.sub(2), Constant(0), fixedBoundary2, method='pointwise')

#bcs = [bcLeft, bcRight, bcFront, bcBack, bcFixed1, bcFixed2]
bcs = [bcLeft, bcRight, bcFront, bcBack, bcBottom]
bcNames = ['Left', 'Right', 'Front', 'Back', 'Bottom']
##########################################################################

# calculate elastic constants
lmbdaF, muF = cal_Lame(Ef, nuF)
lmbdaS, muS = cal_Lame(Es, nuS)

# Lame constants
lmbda = cal_func_KfKs(lmbdaF, lmbdaS, markers, VS)
# shear modulus
mu = cal_func_KfKs(muF, muS, markers, VS)

##########################################################################

# Define variational problem
u, u0, du, v = Function(V), Function(V), TrialFunction(V), TestFunction(V)

Gr = Expression("gF", degree=0, gF=0.0)
Fg = as_matrix(((1. + Gr, 0.0, 0.), (0., 1. + Gr, 0.0), (0., 0.0, 1.0)))

psi = cal_psi_NeoHookean(u, Fg, lmbda, mu)
Psi = psi*dx
GradPsi = derivative(Psi, u, v)
HessPsi = derivative(GradPsi, u, du)

##########################################################################

# doesn't escape the box [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
constraint_u = Expression(("xmax-x[0]", "ymax-x[1]", "zmax-x[2]"),
                          xmax = L, ymax = W, zmax = H + 2.0, degree=1)
constraint_l = Expression(("xmin-x[0]", "ymin-x[1]", "zmin-x[2]"),
                          xmin = 0.0, ymin = 0.0, zmin = 0.0, degree=1)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)
# apply that to the vectors
for idx in range(len(bcs)):
    b = bcs[idx]
    #print ("%s boundary" % bcNames[idx])
    b.apply(u_min.vector())
    b.apply(u_max.vector())
    #print ("%s boundary" % bcNames[idx])
del b

##########################################################################

# Compute solution
steps = 60
step = 0
fileDisp = File('elasticity/displacement.pvd')

bucklingProblem = BucklingProblem(u, Psi, GradPsi, HessPsi)
elasticProblem  = ElasticProblem(GradPsi, HessPsi, bcs)
elasticSolver   = ElasticSolver()
optSolver       = OptSolver()
eigenSolver     = EigenSolver(elasticProblem)

deltaU = 0.0

while step < steps:
    Gr.gF = deltaU
    print ("strain = %f, growth rate = %f" % (deltaU, Gr.gF))
    elasticSolver.solve(elasticProblem, u.vector())
    print ("--------------Eigen Value Solver started--------------------")
    #eigenV, eigenVec = eigen_solver(HessPsi, bcs)
    #eigenSolver      = EigenSolver(elasticProblem)
    #eigenV, eigenVec = eigen_solve(eigenSolver)
    #del eigenSolver
    print("Smallest Eigen Value = %f" % (eigenV))
    print ("--------------Eigen Value Solver ended  --------------------")
    #optSolver.solve(bucklingProblem, u.vector(), u_min.vector(), u_max.vector())
    u0.vector()[:] = eigenVec
    fileDisp << (u0, Gr.gF)
    step += 1
    deltaU = 0.00 + 0.025 * step
##########################################################################
