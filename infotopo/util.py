"""
"""

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

import pandas as pd

from rxnnet.util import Series, Matrix



class DF(pd.DataFrame):
    """
    """
    
    @property
    def _constructor(self, **kwargs):
        return self.__class__
    
    
    @property
    def _constructor_sliced(self):
        return Series



        


def flatten(nested, depth=None):
    """
    """
    raise NotImplementedError



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







