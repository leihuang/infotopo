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
        self.p0_ = pred.p0_
        self.p0 = pred.p0
        self.ptype = pred.ptype
        self.prior = pred.prior

        self.Y = dat.Y
        self.Y_ = dat.Y.values
        self.sigma = dat.sigma
        self.sigma_ = dat.sigma.values

        self.pred = pred
        self.dat = dat

        def r_(p):
            return (self.Y_ - pred.f_(p)) / self.sigma_

        def r(p=None):
            if p is None:
                p = self.p0_
            return Series(r_(p), self.yids)
        
        def Dr_(p):
            return - (pred.Df_(p).T / self.sigma_).T

        def Dr(p=None):
            if p is None:
                p = self.p0_
            else:
                p = np.asarray(p)
            return Matrix(Dr_(p), self.yids, self.pids)
        
        self.r_ = r_
        self.Dr_ = Dr_
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

    def get_cost(self, p=None):
        if p is None:
            p = self.p0_
        else:
            p = np.asarray(p)
        return np.linalg.norm(self.r_(p))**2 / 2


