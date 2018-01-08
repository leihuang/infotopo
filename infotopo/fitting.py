"""
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

import numpy as np
import scipy as sp

from .util import Series, Matrix



class Fit(object):
    """
    """
    def __init__(self, pids, **kwargs):
        assert 'p' in kwargs, 'p has to be provided'
        assert 'cost' in kwargs, 'cost has to be provided'
        assert 'p0' in kwargs, 'p0 has to be provided'

        self.p = kwargs.pop('p')
        self.cost = kwargs.pop('cost')
        self.p0 = kwargs.pop('p0')

        for k, v in kwargs.items():
            setattr(self, k, v)



def fit_nelder_mead(res):
    raise NotImplementedError



def fit_gradient_descent(res):
    raise NotImplementedError



def fit_gauss_newton(res):
    raise NotImplementedError



def fit_lm_scipy(res, p0=None, in_logp=True, **kwargs):
    """Scipy implementation of Levenberg-Marquardt algorithm.
    """
    if p0 is None:
        p0 = res.p0
        
    if in_logp:
        res = res.get_in_logp()
        p0 = np.log(p0)
    
    keys_scipy = ['full_output', 'col_deriv', 'ftol', 'xtol', 'gtol', 
                  'maxfev', 'epsfcn', 'factor', 'diag']
    kwargs = dict([item for item in kwargs.items() if item[0] in keys_scipy])
    kwargs['full_output'] = True

    p, cov, infodict, mesg, ier = sp.optimize.leastsq(func=res._r, x0=p0, 
                                                      Dfun=res._Dr, 
                                                      **kwargs)
    cost = res.cost(p)

    if in_logp:
        p = np.exp(p)
        # FIXME: cov

    r = Series(infodict['fvec'], res.yids)
    covmat = Matrix(cov, res.pids, res.pids)
    nfev = infodict['nfev']
    
    fit = Fit(p=p, cost=cost, p0=p0,
              covmat=covmat, nfev=nfev, 
              r=r, message=mesg, ier=ier, pids=res.pids)
    
    return fit
fit_lm_scipy.__doc__ += sp.optimize.leastsq.__doc__



def fit_lmga(res):
    """Implementation of Levenberg-Marquardt algorithm with geodesic 
    acceleration.
    """
    raise NotImplementedError
