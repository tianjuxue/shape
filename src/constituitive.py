import fenics as fe


def mapped_grad_wrapper(s):
    def mapped_grad(u):
        I = fe.Identity(u.geometric_dimension())
        return fe.dot(fe.grad(u), fe.inv(fe.grad(s) + I))
    if s is None:
        return fe.grad
    else:
        return mapped_grad


def mapped_J_wrapper(s):
    if s is None:
        return 1.
    else:
        I = fe.Identity(s.geometric_dimension())
        return fe.det(fe.grad(s) + I) 


def DeformationGradientFluctuation(grad, v, H):
    I = fe.Identity(v.geometric_dimension())
    grad_u = grad(v) + H      
    return I + grad_u


def DeformationGradient(grad, u):
    I = fe.Identity(u.geometric_dimension())
    return fe.variable(I + grad(u))


def RightCauchyGreen(F):
    return F.T * F


def NeoHookeanEnergyFluctuation(mesh_disp, variable, shear_mod, bulk_mod, fluctuation, H_list=None):
    grad = mapped_grad_wrapper(mesh_disp)
    if fluctuation:
        F = DeformationGradientFluctuation(grad, variable, H_list)
    else:
        F = DeformationGradient(grad, variable)

    F = fe.variable(F)
    J = fe.det(F)
    Jinv = J**(-2 / 3)
    I1 = fe.tr(RightCauchyGreen(F))

    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2) 
 
    first_pk_stress = fe.diff(energy, F)
    constitutive_tensor = fe.diff(first_pk_stress, F)
    sigma_v = von_mises(first_pk_stress, F)
    return energy, first_pk_stress, constitutive_tensor, sigma_v


def von_mises(first_pk_stress, F):
	sigma = 1./fe.det(F) * fe.dot(first_pk_stress, F.T)
	sigma_dev = fe.dev(sigma)
	J2 = 1./2. * fe.inner(sigma_dev, sigma_dev)
	return fe.sqrt(3*J2)


# def epsilon(u):
#     return fe.sym(fe.grad(u))

 
# def sigma(L, u):
#     strain = epsilon(u) 
#     return fe.as_tensor(L[i, j, k, l] * strain[k, l], [i, j])
