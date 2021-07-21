import fenics as fe


def DeformationGradientFluctuation(v, H):
    grad_u = fe.grad(v) + H  
    I = fe.Identity(v.geometric_dimension())
    return I + grad_u


def DeformationGradient(u):
    I = fe.Identity(u.geometric_dimension())
    return fe.variable(I + fe.grad(u))


def RightCauchyGreen(F):
    return F.T * F


def NeoHookeanEnergyFluctuation(variable, young_modulus, poisson_ratio, return_stress, fluctuation, H_list=None):
    shear_mod = young_modulus / (2 * (1 + poisson_ratio))
    bulk_mod = young_modulus / (3 * (1 - 2*poisson_ratio))
    d = variable.geometric_dimension()

    if fluctuation:
        F = DeformationGradientFluctuation(variable, H_list)
    else:
        F = DeformationGradient(variable)

    F = fe.variable(F)
    J = fe.det(F)
    Jinv = J**(-2 / 3)
    I1 = fe.tr(RightCauchyGreen(F))

    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2) 
 
    if return_stress:
        first_pk_stress = fe.diff(energy, F)
        sigma_v = von_mises(first_pk_stress, F)
        return energy, first_pk_stress, sigma_v

    return energy


def von_mises(first_pk_stress, F):
	sigma = 1./fe.det(F) * fe.dot(first_pk_stress, F.T)
	sigma_dev = fe.dev(sigma)
	J2 = 1./2. * fe.inner(sigma_dev, sigma_dev)
	return fe.sqrt(3*J2)



# def strain(grad_u):
#     return 0.5*(grad_u + grad_u.T)


# def psi_linear_elasticity(epsilon, lamda, mu):
#     return lamda / 2 * fe.tr(epsilon)**2 + mu * fe.inner(epsilon, epsilon)
