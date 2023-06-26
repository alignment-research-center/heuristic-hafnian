import numpy as np


def random_symmetric(p):
    """
    Random real symmetric matrix with i.i.d.
    standard normal off-diagonal entries.
    """
    mat = np.random.randn(p, p)
    mat = (mat + mat.transpose()) / 2**0.5
    return mat


def random_complex_symmetric(p):
    """
    Random complex symmetric matrix with i.i.d.
    standard complex normal off-diagonal entries.
    """
    real_mat = np.random.randn(p, p)
    imag_mat = np.random.randn(p, p)
    mat = real_mat + imag_mat * 1j
    mat = (mat + mat.transpose()) / 2
    return mat


def random_hermitian(p):
    """
    Random Hermitian matrix with i.i.d.
    standard complex normal off-diagonal entries.
    """
    real_mat = np.random.randn(p, p)
    imag_mat = np.random.randn(p, p)
    mat = real_mat + imag_mat * 1j
    mat = (mat + mat.conjugate().transpose()) / 2
    return mat


def random_wishart(p, *, cov=None, dof=None):
    """
    Random p x p Wishart matrix with dof degrees of freedom.
    See: https://en.wikipedia.org/wiki/Wishart_distribution
    """
    if cov is None:
        cov = np.eye(p)
    if dof is None:
        dof = p
    assert cov.shape == (p, p)
    samples = np.random.multivariate_normal(np.zeros(p), cov, size=dof)
    return samples.transpose() @ samples


def random_complex_wishart(p, *, cov=None, dof=None):
    """
    Random p x p complex Wishart matrix with dof degrees of freedom.
    See: https://en.wikipedia.org/wiki/Complex_Wishart_distribution
    """
    if cov is None:
        cov = np.eye(p)
    if dof is None:
        dof = p
    real_cov = (
        np.block([[np.real(cov), np.imag(-cov)], [np.imag(cov), np.real(cov)]]) / 2
    )
    real_samples = np.random.multivariate_normal(np.zeros(2 * p), real_cov, size=dof)
    samples = real_samples[:, :p] + real_samples[:, p:] * 1j
    return samples.transpose() @ samples.conjugate()


def random_complex_double_wishart(p, *, cov=None, rel=None, dof=None):
    """
    Random p x p block matrix of the form ((B A) (A* B*)) where
    B is complex symmetric and A is Hermitian positive semi-definite.

    The distribution is analogous to a complex Wishart matrix, except
    instead of being a sample covariance matrix of a complex normal
    distribution, it is a sample block matrix ((rel cov) (cov* rel*))
    where cov is the covariance matrix and rel is the relation matrix
    of a complex normal distribution.
    """
    assert p % 2 == 0
    if cov is None:
        cov = np.eye(p // 2)
    if rel is None:
        rel = np.zeros((p // 2, p // 2))
    if dof is None:
        dof = p // 2
    real_cov = (
        np.block(
            [
                [np.real(cov + rel), np.imag(-cov + rel)],
                [np.imag(cov + rel), np.real(cov - rel)],
            ]
        )
        / 2
    )
    real_samples = np.random.multivariate_normal(np.zeros(p), real_cov, size=dof)
    samples = real_samples[:, : p // 2] + real_samples[:, p // 2 :] * 1j
    psd = samples.transpose() @ samples.conjugate()
    sym = samples.transpose() @ samples
    return np.block([[sym, psd], [psd.conjugate(), sym.conjugate()]])


def random_double_wishart(p, *, cov=None, dof=None):
    """
    Random p x p block matrix of the form ((B A) (A B)) where
    B is real symmetric and A is real symmetric positive semi-definite.

    Equivalent to taking the real part of a complex double Wishart
    with relation matrix zero.
    """
    assert p % 2 == 0
    if cov is None:
        cov = np.eye(p // 2)
    if dof is None:
        dof = p // 2
    samples = np.random.multivariate_normal(np.zeros(p // 2), cov / 2, size=dof * 2)
    mat1 = samples[:dof, :].transpose() @ samples[:dof, :]
    mat2 = samples[dof:, :].transpose() @ samples[dof:, :]
    psd = mat1 + mat2
    sym = mat1 - mat2
    return np.block([[sym, psd], [psd, sym]])
