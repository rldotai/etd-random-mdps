"""
Simple Markov Decision Problems.

- Consider grouping methods together in a class?
"""
import numpy as np 


def unit(ndim, ix, dtype=np.float):
    """Generate a unit vector, with the entry at `ix` set to 1."""
    ret = np.zeros(ndim, dtype=dtype)
    ret[ix] = 1
    return np.atleast_1d(ret)

def rand_p(ns, na):
    """ Generate a random transition probability matrix for the given 
    numbers of states `ns` and actions `na`.
    
    Ensure that each state has some probability of transitioning to a
    different state.
    Consider allowing user to specify random variable to draw from.
    Consider adding code to ensure that the matrix is properly ergodic.
    Consider having an adjustable sparsity parameter.
    """
    ret = np.zeros((ns, na, ns), dtype=np.float)
    for s, a in np.ndindex(ns, na):
        ret[s, a] = np.random.random(ns)
        ret[s, a] = ret[s, a]/np.sum(ret[s, a]) # normalize
    return ret


def prob_next(s, a, pmat):
    """ Get the probability distribution over possible next states, given
    action `a` was taken in state `s`. 
    """
    assert(pmat.ndim == 3)
    ns, na, _ = pmat.shape
    if isinstance(s, int):
        s = unit(ns, s)
    if isinstance(a, int):
        a = unit(na, a)
        
    return np.dot(a, np.dot(s, pmat))


def transition(s, a, pmat):
    """ Transition to a new state according to the probability matrix `pmat`, 
    given action `a` was taken in state `s`.
    """
    assert(pmat.ndim == 3)
    ns, na, _ = pmat.shape
    # Allow specifying states/actions as vectors or integer indices
    if isinstance(s, int):
        s = unit(ns, s)
    if isinstance(a, int):
        a = unit(na, a)
        
    # Select and return the choice
    prob = np.dot(a, np.dot(s, pmat))
    choice = np.random.choice(np.arange(ns), p=prob)
    return unit(ns, choice) 