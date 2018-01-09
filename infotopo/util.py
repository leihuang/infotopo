"""
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np



class Series(pd.Series):
    """
    """

    @property
    def _constructor(self):  
        return Series

        
    def randomize(self, seed=None, distribution='lognormal', **kwargs):
        """
        :param seed:
        :param distribution: name of the distribution used in randomization
            ('lognormal', 'normal', etc.). If 'lognormal': default is sigma=1
        :type distribution: str
        :param kwargs: sigma
        """
        if seed is not None:
            np.random.seed(seed)

        if distribution == 'lognormal':
            if 'sigma' not in kwargs:
                kwargs['sigma'] = 1
            x = self * np.random.lognormal(size=self.size, **kwargs)
        if distribution == 'normal':
            x = self + np.random.normal(size=self.size, **kwargs)
        return x



class DF(pd.DataFrame):
    """
    """
    @property
    def _constructor(self):
        return DF
    
    
    @property
    def _constructor_sliced(self):
        return Series


    @property
    def rowvarids(self):
        return self.index.tolist()


    @property
    def colvarids(self):
        return self.columns.tolist()


    @property
    def nrow(self):
        return self.shape[0]
        

    @property
    def ncol(self):
        return self.shape[1]



class Matrix(DF):
    """
    """
    @property
    def _constructor(self):
        return Matrix


    def __mul__(self, other):
        return Matrix(np.dot(self, other), self.index, other.columns)


    def inv(self):
        return Matrix(np.linalg.inv(self), self.columns, self.index)


    def normalize(self, y=None, x=None):
        """
        M = dy / dx
        M_normed = d logy/d logx = diag(1/y) * M * diag(x)
        """
        mat = self
        if y is not None:
            mat = Matrix.diag(1/y) * mat
        if x is not None:
            mat = mat * Matrix.diag(x)
        return mat 


    @property
    def rank(self):
        return np.linalg.matrix_rank(self)
        

    @staticmethod
    def eye(rowvarids, colvarids=None):
        if colvarids is None:
            colvarids = rowvarids
        return Matrix(np.eye(len(rowvarids)), rowvarids, colvarids)


    @staticmethod
    def diag(x):
        """Return the diagonal matrix of series x. 
        
        D(x) = diag(x)
        """
        return Matrix(np.diag(x), x.index, x.index)



def flatten(nested, depth=None):
    """Flatten a nested sequence by a given depth.

    :param depth: depth of flattening; a depth of 1 means one level down, and
        the default is deepest possible
    """
    nested = list(nested)
    
    if depth is None:
        depth = float('inf')

    d = 1  # current depth

    while d <= depth:
        flattened = []
        for elem in nested:
            if hasattr(elem, '__iter__'):
                flattened.extend(elem)
            else:
                flattened.append(elem)
        if nested == flattened:
            break
        nested = flattened
        d += 1

    flattened = type(nested)(flattened)

    return flattened



def get_product():
    """
    """
    raise NotImplementedError



def sub_expr(s, mapping):
    """
    """
    raise NotImplementedError



def simplify_expr(s):
    """
    """
    raise NotImplementedError



def diff_expr(s):
    """
    """
    raise NotImplementedError







