#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:04:10 2017

@author: shengmao
"""

from __future__ import print_function
from dolfin import *
import ufl
import numpy as np
#from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()


####Material properties
############################################################
# E = 2.0e6; nu = 0.45; rho = 1.0

E = 1e2; nu = 0.3; rho = 1.0

mu = E/(2+2*nu)
lmbda = E*nu/((1+nu)*(1-2*nu))
kV = Constant((0.0,0.0))

####Boundary conditions
############################################################
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Left and Bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary 
        #AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 0.01)) or 
                        (near(x[0], 0.01) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 0.01) and near(x[1], 0.01):
            y[0] = x[0] - 0.01
            y[1] = x[1] - 0.01
        elif near(x[0], 0.01):
            y[0] = x[0] - 0.01
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 0.01

# Fixed boundary at left-bottom edge
tol = 1e-14
def fixed_boundary(x, on_boundary):
    return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)

####Functional definition
############################################################
mesh = Mesh('unit_mesh_20.xml')
V = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary())
bc = DirichletBC(V, Constant((0,0)), fixed_boundary, 
                 method = 'pointwise')
u = TrialFunction(V)
v = TestFunction(V)

P = VectorElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, P*P, constrained_domain=PeriodicBoundary())

def eigen_solver(V,kV):   
    ####Mechanics part
    ############################################################   
    i,j,k,l=ufl.indices(4)
    delta=Identity(2)

    C = as_tensor((lmbda*(delta[i,j]*delta[k,l]) + mu*(delta[i,k]*delta[j,l]+ delta[i,l]*delta[j,k])), (i,j,k,l))
    

    def epsilon(u):
        return sym(grad(u))

    # def sigma(u):
    #     strain = epsilon(u) 
    #     return 2 * mu * strain + lmbda * tr(strain) * Identity(2)

    def sigma(u):
        strain = epsilon(u) 
        return as_tensor(C[i, j, k, l]*strain[k, l], [i, j])

    def ukk(u, kV):
        return as_tensor(C[i, j, k, l]*kV[l]*kV[j]*u[k], [i])

    def uk1(u, kV):
        return as_tensor((C[i,j,k,l]*u[k]*kV[l]),[i, j])
    
    def uk2(u, kV):
        return as_tensor((C[i,j,k,l]*Dx(u[k],l)*kV[j]),[i])
    
    # Define variational problem
    uR, uI = TrialFunctions(V)
    vR, vI = TestFunctions(V)



    # aR = inner(sigma(uR), epsilon(vR))*dx + inner(ukk(uR, kV), vR)*dx - \
    #      inner(uk1(uI, kV), grad(vR))*dx + inner(uk2(uI, kV), vR)*dx
          
    # aI = inner(sigma(uI), epsilon(vI))*dx + inner(ukk(uI, kV), vI)*dx + \
    #      inner(uk1(uR, kV), grad(vI))*dx - inner(uk2(uR, kV), vI)*dx

    # a = aR + aI 

    a = C[i, j, k, l] * (grad(uR)[k, l] * grad(vR)[i, j] + grad(uI)[k, l] * grad(vI)[i, j] + \
                         kV[l] * kV[j] * (uR[k] * vR[i] + uI[k] * vI[i]) + \
                         (uR[k] * grad(vI)[i, j] - uI[k] * grad(vR)[i, j]) * kV[l] - \
                         (vI[i] * grad(uR)[k, l] - vR[i] * grad(uI)[k, l]) * kV[j]) * dx




    m = rho*(inner(uR,vR)*dx + inner(uI,vI)*dx)
    
    # Assemble stiffness form
    A = PETScMatrix()
    M = PETScMatrix()
    assemble(a, tensor=A)
    assemble(m, tensor=M)


    bcs = [DirichletBC(V.sub(0), Constant((0., 0.)), fixed_boundary, method='pointwise'),
           DirichletBC(V.sub(1), Constant((0., 0.)), fixed_boundary, method='pointwise')]

    # for bc in bcs:
    #     bc.apply(A)
    #     bc.apply(M)


    # Create eigensolver
    solver = SLEPcEigenSolver(A, M)
    solver.parameters["solver"] = "krylov-schur"
    solver.parameters["problem_type"] = "gen_hermitian"
    #solver.parameters["spectrum"] = "smallest magnitude"
    solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"
    solver.parameters["spectral_shift"] = 1.0
    neigs = 16
    solver.solve(neigs)
    
    computed_eigenvalues = []
    for i in range(min(neigs, solver.get_number_converged())):
        r, _ = solver.get_eigenvalue(i) # ignore the imaginary part
        computed_eigenvalues.append(r)
        
    o2 = np.sort(np.array(computed_eigenvalues))

    freq = np.sqrt(abs(o2))/2./pi
    
    return freq



bandgaps = []
kts = np.linspace(0,3,31)
for kt in kts:
    if (kt<1):
        kx = (pi/0.01)*kt
        ky = 0.0
    elif (kt<2):
        kx = (pi/0.01)
        ky = (kt-1.0)*(pi/0.01)
    else:
        kx = (3-kt)*(pi/0.01)
        ky = kx
    kV = Constant((kx,ky))
    freqt = eigen_solver(V,kV)
    bandgaps.append(freqt)

bArray = np.array(bandgaps)
plt.plot(bArray/1000., marker='o')


vtkfile = File(f'mesh.pvd')
vtkfile << mesh

print(bArray) # 31x16
print(bArray.shape)
plt.show()

#o = np.sqrt(o2)
## Compute all eigenvalues of A x = \lambda x
#print("Computing eigenvalues. This can take a minute.")
#eigensolver.solve()
#
## Extract largest (first) eigenpair
#r, c, rx, cx = eigensolver.get_eigenpair(0)
#
#print("Largest eigenvalue: ", r)
#
## Initialize function and assign eigenvector
#u = Function(V)
#u.vector()[:] = rx
#
## Plot eigenfunction
#plot(u)
#interactive()
