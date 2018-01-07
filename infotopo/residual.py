"""
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from infotopo.util import Series, Matrix


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
    
    
    def __call__(self, p=None):
        return self.r(p=p)


