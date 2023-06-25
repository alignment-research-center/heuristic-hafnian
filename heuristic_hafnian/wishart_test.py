import numpy as np

from .wishart import (
    random_wishart,
    random_complex_wishart,
    random_complex_double_wishart,
    random_double_wishart,
)


def test_random_wishart():
    np.random.seed(0)
    dof = 1000
    for p in range(1, 20):
        cov = random_wishart(p)
        assert np.isclose(cov, cov.transpose()).all()
        assert (np.linalg.eigvals(cov) > 0).all()
        err = random_wishart(p, cov=cov, dof=dof) / dof - cov
        err[np.arange(p), np.arange(p)] = np.diag(err) / dof**0.5
        mse = (err**2).mean() * dof / p**2
        assert mse < 2.5


def test_random_complex_wishart():
    np.random.seed(0)
    dof = 1000
    for p in range(1, 20):
        cov = random_complex_wishart(p)
        assert np.isclose(cov, cov.conjugate().transpose()).all()
        assert (np.linalg.eigvals(cov) > 0).all()
        err = random_complex_wishart(p, cov=cov, dof=dof) / dof - cov
        err[np.arange(p), np.arange(p)] = np.diag(err) / dof**0.5
        mse = (np.abs(err) ** 2).mean() * dof / p**2
        assert mse < 2.5


def test_random_complex_double_wishart():
    np.random.seed(0)
    dof = 1000
    for p in range(2, 20, 2):
        mat = random_complex_double_wishart(p)
        assert np.isclose(
            mat[: p // 2, : p // 2], mat[p // 2 :, p // 2 :].conjugate()
        ).all()
        assert np.isclose(
            mat[: p // 2, p // 2 :], mat[p // 2 :, : p // 2].conjugate()
        ).all()
        cov = mat[: p // 2, p // 2 :]
        rel = mat[: p // 2, : p // 2]
        assert np.isclose(cov, cov.conjugate().transpose()).all()
        assert (np.linalg.eigvals(cov) > 0).all()
        assert np.isclose(rel, rel.transpose()).all()
        err = random_complex_double_wishart(p, cov=cov, rel=rel, dof=dof) / dof - mat
        err[np.arange(p), np.arange(p)] = np.diag(err) / dof**0.5
        err[np.arange(p // 2), np.arange(p // 2, p)] = (
            err[np.arange(p // 2), np.arange(p // 2, p)] / dof**0.5
        )
        err[np.arange(p // 2, p), np.arange(p // 2)] = (
            err[np.arange(p // 2, p), np.arange(p // 2)] / dof**0.5
        )
        mse = (np.abs(err) ** 2).mean() * dof / (p // 2) ** 2
        assert mse < 2.5


def test_random_double_wishart():
    np.random.seed(0)
    dof = 1000
    for p in range(2, 20, 2):
        mat = random_double_wishart(p)
        assert np.isclose(mat[: p // 2, : p // 2], mat[p // 2 :, p // 2 :]).all()
        assert np.isclose(mat[: p // 2, p // 2 :], mat[p // 2 :, : p // 2]).all()
        cov = mat[: p // 2, p // 2 :]
        rel = mat[: p // 2, : p // 2]
        assert np.isclose(cov, cov.transpose()).all()
        assert (np.linalg.eigvals(cov) > 0).all()
        assert np.isclose(rel, rel.transpose()).all()
        sample = random_double_wishart(p, cov=cov, dof=dof)
        err = sample[: p // 2, p // 2 : p] / dof - cov
        err[np.arange(p // 2), np.arange(p // 2)] = np.diag(err) / dof**0.5
        mse = 2 * (err**2).mean() * dof / (p // 2) ** 2
        assert mse < 2.5
