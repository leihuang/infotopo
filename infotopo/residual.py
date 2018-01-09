"""
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from infotopo.util import Series, Matrix



class Residual(object):
    """
    """
    def __init__(self, pred, dat):
        """
        :param pred:
        :param dat: Y (mean) and sigma
        """
        self.pids = pred.pids
        self.yids = pred.yids
        self.pdim = pred.pdim
        self.ydim = pred.ydim
        self._p0 = pred._p0
        self.p0 = pred.p0
        self.ptype = pred.ptype
        self.prior = pred.prior

        self.Y = dat.Y
        self._Y = dat.Y.values
        self.sigma = dat.sigma
        self._sigma = dat.sigma.values

        self.pred = pred
        self.dat = dat

        def _r(p):
            return (self._Y - pred._f(p)) / self._sigma

        def r(p=None):
            if p is None:
                p = self._p0
            return Series(_r(p), self.yids)
        
        def _Dr(p):
            return - (pred._Df(p).T / self._sigma).T

        def Dr(p=None):
            if p is None:
                p = self._p0
            return Matrix(_Dr(p), self.yids, self.pids)
        
        self._r = _r
        self._Dr = _Dr
        self.r = r
        self.Dr = Dr
        
    
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


