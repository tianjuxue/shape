import jax.numpy as np
import jax
import matplotlib.pyplot as plt


def taylor_test():
    t = 0.5
    h = 1e-5

    def get_A(t):
        A = np.array([[t, np.cos(t), np.exp(t)], [t**2, t**3, np.sin(t)], [2*t, t + 1, 3*t + 2]])
        return 0.5*(A + A.T)

    grad_A = jax.jacrev(get_A)

    dA = grad_A(t)
    A = get_A(t)
    w, v = jax.numpy.linalg.eig(A)
    f = w[0]
    u = v[:, 0]
    print(np.real(w))
    print(np.real(v))
    dJdm = np.dot(np.dot(dA, u), u) / np.dot(u, u)

    A_h = get_A(t + h)
    w, v = jax.numpy.linalg.eig(A_h)
    print(np.real(w))
    print(np.real(v))
    f_h = w[0]

    A_2h = get_A(t + 2*h)
    print(np.real(w))
    print(np.real(v))
    w, v = jax.numpy.linalg.eig(A_2h)
    f_2h = w[0]

    f = np.real(f)
    dJdm = np.real(dJdm)
    f_h = np.real(f_h)
    f_2h = np.real(f_2h)


    def lamb(t):
        A = get_A(t)
        w = jax.numpy.linalg.eigvals(A)
        f = np.real(w[0])
        return f

    dJdm_jax = jax.grad(lamb)(t)
    print(f"f = {f}, formula dJdm = {dJdm}, jax autograd gives: {dJdm_jax}")
    print(f"f_h - f = {f_h - f}, f_2h - f = {f_2h - f}")
    print(f"r_h = {f_h - f - np.dot(dJdm_jax, h)}, r_2h = {f_2h - f - np.dot(dJdm_jax, 2*h)}")
    print(f"finite difference: {(f_h - f) / h}")

    tt = np.linspace(0.3, 0.6, 101)
    ff = []
    for t in tt:     
        A = get_A(t)
        w, v = jax.numpy.linalg.eig(A)
        f = w[0]
        ff.append(f)

    plt.plot(tt, ff, marker='o')
    plt.show()


if __name__ == '__main__':
    taylor_test()