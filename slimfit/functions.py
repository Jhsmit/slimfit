from __future__ import annotations

import typing

import numpy as np
import sympy as sp
from sympy import Matrix, zeros, Symbol, Expr

from slimfit import MatrixNumExpr

if typing.TYPE_CHECKING:
    from slimfit.symbols import Variable


def gaussian_numpy(
    x: np.array | float, mu: np.array | float, sig: np.array | float
) -> np.array | float:
    return 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2) / 2)


def gaussian_sympy(x: Symbol, mu: Symbol, sig: Symbol) -> Expr:
    return sp.exp(-((x - mu) ** 2) / (2 * sig ** 2)) / (sp.sqrt(2 * np.pi) * sig)


def gaussian(
    x: Variable, mu: Matrix | MatrixNumExpr, sigma: Matrix | MatrixNumExpr
) -> Matrix | MatrixNumExpr:
    if mu.shape != sigma.shape:
        raise ValueError("Shape mismatch between 'mu' and 'sigma'")

    m_out = zeros(*mu.shape)
    for i, j in np.ndindex(mu.shape):
        m_out[i, j] = gaussian_sympy(x, mu[i, j], sigma[i, j])

    if isinstance(mu, MatrixNumExpr) or isinstance(sigma, MatrixNumExpr):
        return MatrixNumExpr(m_out, kind="GMM")
    else:
        return m_out


def gaussian_matrix(x: Variable, mu: Matrix, sigma: Matrix) -> Matrix:
    """Applies gaussian function elementwise from a 'mu' and 'sigma' matrix; given a variable 'x'"""
    if mu.shape != sigma.shape:
        raise ValueError("Shape mismatch between 'mu' and 'sigma'")

    m_out = zeros(*mu.shape)
    for i, j in np.ndindex(mu.shape):
        m_out[i, j] = gaussian_sympy(x, mu[i, j], sigma[i, j])

    return m_out
