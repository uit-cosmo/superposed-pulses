import numpy as np


def _sample_asymm_laplace(scale=1.0, shape=0.5, size=1, seed=None):
    """
    Use:
        sample_asymm_laplace(scale=1., shape=0.5, size=1, seed=None)
    Random samples drawn from the asymmetric Laplace distribution
    using inverse sampling. The distribution is given by
    F(A;alpha,kappa) = 1-(1-kappa)*Exp(-A/(2*alpha*(1-kappa))), A>0
                       kappa*Exp[A/(2*alpha*kappa)], A<0
    where F is the CDF of A, alpha is a scale parameter and
    kappa is the asymmetry parameter.
    Input:
        scale: scale parameter alpha. ................... float, scale>0
        shape: shape (asymmetry) parameter kappa. ....... float, 0<=shape<=1
        size: number of points to draw. 1 by default. ... int, size>0
        seed: specify a random seed. .................... int
    Output:
        X: Array of randomly distributed values. ........ (size,) np array
    """

    assert scale > 0.0
    assert (shape >= 0.0) & (shape <= 1.0)
    prng = np.random.RandomState(seed=seed)
    U = prng.uniform(size=size)
    X = np.zeros(size)
    X[U > shape] = -2 * scale * (1 - shape) * np.log((1 - U[U > shape]) / (1 - shape))
    X[U < shape] = 2 * scale * shape * np.log(U[U < shape] / shape)

    return X
