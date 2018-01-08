"""
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

import numpy as np

from .util import Series, Matrix



class Residual(object):
    """
    """
    def __init__(self, pred, dat):
        """
        :param pred:
        :param dat: Y (mean) and sigma
        """
        p0, pids, yids = pred.p0, pred.pids, pred.yids
        Y, sigma = dat.Y.values, dat.sigma.values

        def _r(p):
            return (Y - pred._f(p)) / sigma

        def r(p=None):
            if p is None:
                p = p0
            return Series(_r(p), yids)
        
        def _Dr(p):
            return - (pred._Df(p).T / sigma).T

        def Dr(p=None):
            if p is None:
                p = p0
            return Matrix(_Dr(p), yids, pids)
        
        self._r = _r
        self._Dr = _Dr
        self.r = r
        self.Dr = Dr
        self.pids = pids
        self.yids = yids
        self.p0 = p0
        self.pred = pred
        self.dat = dat
        self.ptype = pred.ptype
    

    
    def __call__(self, p=None):
        return self.r(p=p)



    def get_in_logp(self):
        """
        """
        assert self.ptype == '', "residual not in bare parametrization"
        pred, dat = self.pred, self.dat
        pred_logp = pred.get_in_logp()
        return Residual(pred_logp, dat)



    def cost(self, p=None):
        return np.linalg.norm(self(p))**2 / 2


