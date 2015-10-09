"""
Helpful functions for working with Markov Chains and Markov Decision Problems.
"""

import numpy as np
import scipy.stats
from functools import reduce 
from numpy.linalg import det, pinv, matrix_rank, norm

# General Linear Algebra 

def mult(*arrays):
    """Array multiplication for a sequence of arrays."""
    return reduce(np.dot, arrays)

def cols(mat):
    assert(is_matrix(mat))
    return [x for x in mat.T]

def colsum(mat):
    assert(is_matrix(mat))
    return np.sum(mat, axis=1)

def rows(mat):
    assert(is_matrix(mat))
    return [x for x in mat]

def rowsum(mat):
    assert(is_matrix(mat))
    return np.sum(mat, axis=0)

def reduce_rank(mat, n):
    """Reduce a matrix's rank to (at most) `n` using SVD."""
    assert(is_matrix(mat))
    U, S, V = np.linalg.svd(mat)
    S[n:] = 0
    ret = np.dot(U, np.dot(np.diag(S), V))
    ret[np.isclose(ret,0)] = 0
    return ret

def someone(n, *zeros):
    """Generate a vector of ones except at particular indices."""
    # TODO: get a better name for this!
    # TODO: fix issue where *zeros is empty
    ret = np.ones(n)
    ret[zeros] = 0
    return ret

def somenone(n, *ones):
    """Generate a vector of zeros except for ones at the given indices."""
    # TODO: a better name for this!
    ret = np.zeros(n)
    ret[ones] = 1
    return ret

# Probability Distributions

def is_probability(pvec, tol=1e-6):
    """Check if a vector represents a probability distribution."""
    vec = np.ravel(pvec)
    return (np.size(vec) == np.size(pvec)) \
    and (np.all(vec >= 0)) \
    and (1-tol <= np.sum(vec) <= 1+tol)

def normalize(array, axis=None):
    """Normalize an array along an axis."""
    def _normalize(vec):
        return vec/np.sum(vec)
    if axis:
        return np.apply_along_axis(_normalize, axis, array)
    else:
        return _normalize(array)


# Stochastic Matrices

## Properties of stochastic matrices

def is_matrix(mat):
    """Test that an array is a matrix."""
    return mat.ndim == 2

def is_square(mat):
    """Ensure that an array is a 2-D square matrix."""
    return (mat.ndim == 2) and (mat.shape[0] == mat.shape[1])

def is_diagonal(mat):
    """Check if a matrix is diagonal."""
    if not is_square(mat):
        return False
    else:
        off_diagonals = np.extract(1 - np.eye(len(mat)), mat)
        return np.all(0 == off_diagonals)

def is_nonnegative(mat):
    """Check if a matrix is nonnegative."""
    return np.all(mat > 0)

def is_stochastic(mat, tol=1e-6):
    """Check if a matrix is (right) stochastic."""
    return (mat.ndim == 2) \
    and (mat.shape[0] == mat.shape[1]) \
    and (np.all([row >= 0 for row in mat])) \
    and (all(1-tol <= np.sum(row) <= 1+tol for row in mat))

def is_substochastic(mat, tol=1e-6):
    """Check if a matrix is (right) substochastic."""
    return (mat.ndim == 2) \
    and (mat.shape[0] == mat.shape[1]) \
    and (np.all([row >= 0 for row in mat])) \

def is_absorbing(mat):
    """Check if the transition matrix has absorbing states."""
    return len(find_terminals(mat)) > 0

def is_periodic(mat):
    """Check if the transition matrix is periodic."""
    return (1 < get_period(mat))

def is_recurrent(mat):
    """Check if the matrix is recurrent."""
    # TODO: Could be tighter, see 54-8 in Handbook of Linear Algebra
    return all(np.all(0 < x) for x in get_all_stationary(mat))

def is_reducible(mat):
    """Check if the matrix is reducible. """
    #TODO: Find a better method for this
    P = np.copy(mat)
    S = np.zeros_like(mat)
    for i in range(len(mat)):
        S += P 
        P = np.dot(P, mat)
    return np.any(np.isclose(0,S))

def is_ergodic(mat):
    """Check if the matrix is ergodic (irreducible and aperiodic)."""
    return not(is_reducible(mat) and is_periodic(mat))


## More general utilities

def find_terminals(mat):
    """Find terminal states in a transition matrix."""
    return [row for ix, row in enumerate(mat) if row[ix] == 1]

def find_nonterminals(mat):
    """Find nonterminal states in a transition matrix."""
    return [row for ix, row in enumerate(mat) if row[ix] != 1]

def find_terminal_indices(mat):
    """Find indices of terminal (absorbing) states in a transition matrix."""
    return [ix for ix, row in enumerate(mat) if row[ix] == 1]

def get_period(mat):
    """Find the period, assuming that stochastic matrix `mat`
    is irreducible (states form a single communicating class).
    
    NB: There is probably a better way of doing this
    """ 
    from fractions import gcd
    from functools import reduce
    tmp = []
    P = np.eye(len(mat))
    for i in range(len(mat)):
        P = np.dot(P, mat)
        if np.any(np.diag(P) > 0):
            tmp.append(i)
    return reduce(gcd, tmp)


def approx_stationary(mat, s0=None, tol=1e-6, iterlimit=10000):
    assert(is_stochastic(mat,tol))
    if s0 is None:
        s0 = np.ones(len(mat))
        s0 = s0/np.sum(s0)
    assert(is_probability(s0, tol))
    
    # Approximate the stationary distribution by repeated transitions
    s = np.copy(s0)
    for i in range(iterlimit):
        sp = np.dot(s, mat)
        if np.allclose(s, sp):
            return s
        s = sp
    else:
        raise Exception("Failed to converge within tolerance:", s, s0)


def stationary(mat):
    """Compute the stationary distribution for transition matrix `mat`, via 
    computing the solution to the system of equations (P.T - I)*\pi = 0. 
        
    NB: Assumes `mat` is ergodic (aperiodic and irreducible).
    Could do with LU factorization -- c.f. 54-14 in Handbook of Linear Algebra
    """
    assert(is_stochastic(mat))
    
    P = (np.copy(mat).T - np.eye(len(mat)))
    P[-1,:] = 1
    b = np.zeros(len(mat))
    b[-1] = 1
    x = np.linalg.solve(P, b)
    return normalize(x)


def get_all_stationary(mat):
    """Compute /all/ stationary states for transition matrix `mat`, by 
    finding the left eigenvectors with an associated eigenvalue of `1`. 
    
    NB: Has a lot of transposing going on in order to accomodate numpy.
    NB: Uses `np.isclose` for checking whether eigenvalues are 1.
    NB: Tries to ensure it returns real-valued left eigenvectors.
    """
    assert(is_stochastic(mat))
    P = np.copy(mat).T
    vals, vecs = np.linalg.eig(P)
    states = [v/np.sum(v) for e, v in zip(vals, vecs.T) if np.isclose(e,1)]
    return [np.real_if_close(v) for v in states]


def distribution_matrix(mat):
    """Compute the stationary distribution for a matrix, and return the 
    diagonal matrix with the stationary distribution along its diagonal."""
    return np.diag(stationary(mat))


# Generating stochastic matrices and MDPs

def random_binary(nrows, ncols, row_sum):
    """Generate a random binary matrix ."""
    tmp = np.zeros(ncols)
    tmp[:row_sum] = 1
    ret = np.zeros((nrows, ncols))
    for i in range(nrows):
        ret[i] = np.random.permutation(tmp)
    return ret


def transition_matrix(ns, pvar=None):
    """ Generate a random ergodic transition matrix with `ns` total states. """
    if pvar is None:
        pvar = scipy.stats.uniform()
    ret = np.zeros((ns,ns))
    for s in range(ns):
        ret[s] = np.abs(pvar.rvs(ns))
        ret[s] = ret[s]/np.sum(ret[s])
    return ret


def random_mdp(ns, na, pvar=None, rvar=None):
    """ Generate a random MDP with `ns` total states, `na` total 
    actions, returning a transition matrix and reward matrix, `(P, R)`.
    
    Optionally, the random variables used to define the probability and
    reward matrices can be specified (c.f. `scipy.stats` )
    """
    if rvar is None:
        rvar = scipy.stats.uniform()
    if pvar is None:
        pvar = scipy.stats.uniform()
        
    P = np.zeros((ns, na, ns), dtype=np.float)
    R = np.zeros((ns, na, ns), dtype=np.float)
    for s, a in np.ndindex(ns, na):
        R[s, a] = rvar.rvs()
        P[s, a] = pvar.rvs(ns)
        P[s, a] = P[s, a]/np.sum(P[s, a])
    return P, R