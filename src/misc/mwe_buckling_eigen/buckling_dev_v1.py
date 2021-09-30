#!/usr/bin/env python
# coding: utf-8
"""
Created Fri Sep 10 16:01:15 CST 2021
@author: ShengMao

This file uses eigen solver to solve the buckling problem:
solve: {(u, g), F(u, g) == 0},
"""
######################################################################################
# import pkgs
######################################################################################

# standard pkgs from the envs
import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import os; import sys
# in-house code
from HyperElastic_Buckling import NearlyIncompElasticity
from DRSolve import dynamic_relaxation_solve
# parameter settings
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["cpp_optimize_flags"] = '-O2 -funroll-loops'
dl.parameters["form_compiler"]["optimize"] = True
dl.parameters["form_compiler"]["quadrature_degree"] = 2 # number of integration points
dl.parameters["form_compiler"]["representation"] = "uflacs"
dl.parameters["linear_algebra_backend"] = "PETSc"
# other settings
plt.rcParams.update({'font.size': 15}) # set the font size
tol = 1e-6 # tolerance for boundary identification
np.random.seed(0) # use the same random seeds

#####################################################################################
# set the geometry accordingly
#####################################################################################

#def AssembleBCs(V, boundary_markers):
#   return [dl.DirichletBC(V, (0.0, 0.0), boundary_markers, 1)]


# define boundaries
def corner(x, on_boundary):
    return dl.near(x[0], 0., 1e-5) and dl.near(x[1], 0.0, 1e-5)

# define boundaries
def left(x, on_boundary):
    return dl.near(x[0], 0., 1e-5)


def AssembleBCs(V, right):
    return [dl.DirichletBC(V.sub(0), (0.0), left),
            dl.DirichletBC(V.sub(0), (0.0), right),
            dl.DirichletBC(V.sub(1), (0.0), corner, "pointwise")]


def CreateFg(growthFactor, type="uniaxial"):
    gf = growthFactor
    if type == "uniaxial":
        Fg = dl.as_matrix(((1.0 + gf, 0.0), (0., 1.0)))
    elif type == "isotropic":
        Fg = dl.as_matrix(((1.0 + gf, 0.0), (0., 1.0 + gf)))
    else:
        Fg = None
        print ("Error specifying growth type")
        sys.exit()
    return Fg


# eigen solver to determine the wavelength
def eigen_solver(A, V, J, bcs):
    # Assemble stiffness form
    v = dl.TestFunction(V)
    dummy = v[0]*dl.dx
    dl.assemble_system(J, dummy, bcs, A_tensor=A)
    
    # calculating the eigen values
    # Create eigensolver
    solver = dl.SLEPcEigenSolver(A)
    solver.parameters["solver"] = "krylov-schur"
    solver.parameters["tolerance"] = 1e-9
    solver.parameters["problem_type"] = "hermitian"
    solver.parameters["spectrum"] = "target real"
    #solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"
    solver.parameters["spectral_shift"] = -5.0
    neigs = 10
    solver.solve(neigs)
    del v
    
    try:
        r, _ , u, _ = solver.get_eigenpair(0)
    except Exception:
        r = 1.0; u = 0.
    
    return r, u


def eigen_solver1(A, V, J, bcs):
    # Assemble stiffness form
    v = dl.TestFunction(V)
    u = dl.TrialFunction(V)
    dummy = v[0]*dl.dx
    mass  = dl.inner(u, v)*dl.dx
    dl.assemble_system(J, dummy, bcs, A_tensor=A)
    M = dl.PETScMatrix()
    dl.assemble(mass, tensor=M)
    # calculating the eigen values
    # Create eigensolver
    solver = dl.SLEPcEigenSolver(A, M)
 
    solver.parameters["solver"] = "krylov-schur"
    solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"

    solver.parameters["spectral_shift"] = -1e-6
    neigs = 100
    solver.solve(neigs)
    #del v
    

    r, _ , u, _ = solver.get_eigenpair(0)
    
    return r, u
    # return solver


if __name__ == '__main__':

    dl.set_log_level(dl.LogLevel.ERROR);
    pi = 3.14159; length = 2*pi*5
    mesh = dl.RectangleMesh(dl.Point(0, 0), dl.Point(length, 1), 10, 1)
    V = dl.VectorFunctionSpace(mesh, "CG", degree=2)
    
    # define boundaries
    def right(x, on_boundary):
        return dl.near(x[0], length, 1e-5)


    u = dl.Function(V); uSave = dl.PETScVector(); uSave.init(V.dim()); 
    u0 = dl.Function(V); A = dl.PETScMatrix();
    bcs = AssembleBCs(V, right); bcs_homo = AssembleBCs(V, right);
    E = 3.0; nu = 0.50; mu = E/2/(1+nu); 
    growthFactor = dl.Constant(1e-4); dg = 1e-4;
    Fg = CreateFg(growthFactor, "uniaxial"); finalGrowth = 1e-2;
    fileU = dl.File("buckling_data/displacement.pvd")
    datafolder = "./buckling_data/"

    F, J =  NearlyIncompElasticity(mu, u, Fg)
    problem = dl.NonlinearVariationalProblem(F, u, bcs, J)
    solver = dl.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    solver.parameters['newton_solver']['maximum_iterations'] = 50
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'

    growth_cur = float(growthFactor)
    while  growth_cur < finalGrowth:
            growth_cur += dg
            growthFactor.assign( growth_cur )
            uSave[:] = u.vector()

            conv_dr = False; conv_newton = False;

            try:
                nIters, conv_newton = solver.solve()
            except RuntimeError:
                u.vector()[:] = uSave[:]
                nIter, conv_dr = dynamic_relaxation_solve(F, u, J, bcs, bcs_homo, 
                    tol=1e-8)

            if conv_dr or conv_newton:
                r, uEigen = eigen_solver1(A, V, J, bcs)
                fileU << (u, growth_cur)
                print("growth = %g, dg = %g, r = %g" % (growth_cur, dg, r))
                if r < 1e-8:
                    np.save(datafolder + "eigen_vector.npy", uEigen)
                    break
            else:
                break

    u0.vector()[:] = uEigen
    Npts = 201; ux = np.zeros(Npts); uy = np.zeros(Npts)
    xLine = np.linspace(0, length, Npts)
    for i in range(Npts):
        ux[i], uy[i] = u0(xLine[i], 1./2.)
        
    plt.plot(xLine, uy, 'k.'); plt.show()



