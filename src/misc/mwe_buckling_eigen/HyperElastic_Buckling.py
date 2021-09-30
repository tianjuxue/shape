#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created Tue Dec 18 22:40:44 EST 2018
@author: ShengMao

this file contains different models for hyperelasticity

"""


# standard pkgs from the envs
import dolfin as dl

# nearly incompressible material, kappa/mu > 10.0
def NearlyIncompElasticity(mu, u, Fg):
	
	dim = u.geometric_dimension()
	V = u.function_space()
	u_, du = dl.TestFunction(V), dl.TrialFunction(V)
	I = dl.Identity(dim)
	ln = dl.ln
	Ft = I + dl.grad(u); F = Ft * dl.inv(Fg)
	J = dl.det(F)
	psi = mu/2 * (dl.tr(F.T*F) + 1/J**2 - 3)
	Pi = psi*dl.dx

	Res = dl.derivative(Pi, u, u_)
	Jac = dl.derivative(Res, u, du)

	return Res, Jac
