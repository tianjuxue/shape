#!/usr/bin/env python
# coding: utf-8
"""
Created Tue Dec 18 20:42:05 EST 2018
@author: ShengMao

This file creates a serial code for dynamic relaxation based on: PhD thesis of
D. J. Luet, 2016.
"""
################################################################################
# import pkgs and global seetings
################################################################################

# standard pkgs
import dolfin as dl
import numpy as np
import scipy.sparse as sp
from math import sqrt
import time # measurements of time
# import warnings
# from mpi4py import mpi_comm_world
#warnings.simplefilter("ignore",sp.SparseEfficiencyWarning)
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["cpp_optimize_flags"] = '-O2 -funroll-loops'
dl.parameters["form_compiler"]["optimize"] = True
dl.parameters["form_compiler"]["quadrature_degree"] = 2 # number of integration points
dl.parameters["form_compiler"]["representation"] = "uflacs"
dl.parameters["linear_algebra_backend"] = "PETSc"

################################################################################
# dynamic relaxation
################################################################################

def dynamic_relaxation_solve(F, u, J, bc=None, bc_homo=None,
                            tol = 1e-9,  CalcCof = 40,
                            info = False, info_force = True):
    
    # default parameters
    iprint = 10; tol_R = tol # do not mess up with tolR
    cmin  = 1e-3; cmax  = 3.9; h_tilde = 1.10; h = 1.;
    
    
    timeZ = time.time()
    # calculate total dofs
    N = u.vector().local_size(); 
    print("----------number of DOF's: %d-------------" % N)
    
    if bc:
        for b in bc:
            b.apply(u.vector())
    
    # initialize displacements, velocities and accelerations at current time step
    q = u.vector()[:]; qdot=np.zeros(N); qdotdot=np.zeros(N);
    # initialize displacements, velocities and accelerations from a previous time step
    q_old=np.zeros(N); qdot_old=np.zeros(N); qdotdot_old=np.zeros(N);

    # initialize forces
    RA = dl.assemble(F); R = np.zeros(N); R_old = np.zeros(N);
    if bc_homo:
        for b in bc_homo:
            b.apply(RA)
    R[:] = RA.get_local(); R = R.squeeze();

    # assemble for Jacobian
    KMat = dl.EigenMatrix(N, N); dl.assemble(J, tensor=KMat);
    if bc:
        for b in bc: b.apply(KMat)
    row,col,val = KMat.data(); K = sp.csr_matrix((val,col,row));

    # calculate the mass matrix with Eqn 4.119 in Luet 2006
    M = h_tilde*h_tilde/4. * np.array(np.absolute(K).sum(axis = 1)).squeeze()

    # compute initial velocity
    qdot[:] = - h/2. * R / M;
    if info_force:
        print('------------------------------------')
        print('Max force: ',np.max(np.absolute(R)))
        print('Max velocity: ',np.max(np.absolute(qdot)))
        print('Max acceleration: ',np.max(np.absolute(qdotdot)))

    ii = 0; iK = 0; 

    while np.max(np.absolute(R))>tol_R:
        ii+=1; iK+=1;
        #  save previous step
        q_old[:]=q[:]; R_old[:]=R[:];
        # step forward
        q[:] += h * qdot; u.vector()[:] = q[:];     
        dl.assemble(F, tensor = RA);
        if bc_homo:
            for b in bc_homo: b.apply(RA)
        R[:] = RA.get_local(); R = R.squeeze();

        # rayleigh quotient to estimate eigenvalues
        S0 = np.dot((R - R_old)/h,  qdot); t = S0/np.einsum('i,i,i', qdot,M,qdot)
        
        if ii%iprint==1:
            #print('S0: ', S0, 'M0: ', np.einsum('i,i,i', qdot,M,qdot))
            if info==True: print('Damping t: ',t);
        if t<0.: t=0.
        c=2.*sqrt(t)
        if (c<cmin): c = cmin
        if (c>cmax): c = cmax
        if ii%iprint==1: 
            if info==True: print('Damping coefficient: ',c)

        eps =h_tilde*h_tilde/4. * np.absolute(
             np.divide((qdotdot - qdotdot_old ), (q - q_old ), 
                        out=np.zeros_like((qdotdot - qdotdot_old )), 
                        where=(q - q_old )!=0))

        if ii%iprint==1: 
            if info==True: print('Max epsilon: ',np.max(eps))
        if ((np.max(eps)>1) and (iK>CalcCof)): #SPR JAN max --> min
            iK=0;
            if info==True: print('Recalculating the tangent matrix: ',ii)
            dl.assemble(J, tensor=KMat);
            if bc:
                for b in bc: b.apply(KMat)
            row,col,val = KMat.data(); K = sp.csr_matrix((val,col,row));
            M[:] = h_tilde*h_tilde/4. * np.array(np.absolute(K).sum(axis = 1)).squeeze()

        qdot_old[:] = qdot[:]
        qdotdot_old[:] = qdotdot[:]

        qdot =(2.-c*h)/(2+c*h)*qdot_old - 2.*h / (2. + c*h) * R / M 
        qdotdot = (qdot - qdot_old) / h

        if ii%iprint == 1 and info_force:
            print('------------------------------------')
            print('Max force: ',np.max(np.absolute(R)))
            print('Max velocity: ',np.max(np.absolute(qdot)))
            print('Max acceleration: ',np.max(np.absolute(qdotdot)))

    u.vector()[:] = q[:]
    timeK = time.time()
    print("Loop time: ", timeK-timeZ)
    
    convergence = True
    if np.isnan(np.max(np.absolute(R))):
        convergence = False

    if convergence:
        print("-------DRSolve successful! converged in %d steps-----------------" % (ii))
    else:
        print("-------FAILED to converged-----------------")

    return ii, convergence




