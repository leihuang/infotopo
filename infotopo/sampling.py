"""
"""

from __future__ import absolute_import, division, print_function
import time
from collections import OrderedDict as OD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from infotopo import util

import imp
imp.reload(util)



class Ensemble(util.DF):

    @property
    def _constructor(self):
        return Ensemble


    @property
    def acceptance_ratio(self):
        return (ens.p.drop_duplicates().shape[0]-1) / (ens.shape[0]-1)


    def scatter(self, hist=False, log10=False, pts=None, colors=None,
                figsize=None, adjust=None, labels=None, labelsize=6,
                nodiag=False, lims=None,
                filepath='', show=True):
        """
        :param hist: if True, also plot histograms for the marginal 
            distributions
        :param pts: a list of points
        :param colors: a str or a list
        :param filepath:
        """
        n = self.ncol

        if figsize is None:
            figsize = (n*2, n*2)
        if labels is None:
            labels = self.colvarids
        if pts is not None:
            if colors is None:
                colors = ['r'] * len(pts)
            elif isinstance(colors, str):
                colors = [colors] * len(pts)
            else:
                raise ValueError

        fig = plt.figure(figsize=figsize)

        n = self.ncol

        if n == 1:
            raise ValueError("Cannot do scatterplot with 1d data.")

        if n == 2:
            ax = fig.add_subplot(111)
            x, y = self.iloc[:,0], self.iloc[:,1]
            ax.scatter(x, y, s=1)
            if pts is not None:
                for pt, c in zip(pts, colors):
                    ax.scatter(*pt, marker='o', color=c, s=10)  
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())            
            if log10:
                ax.set_xscale('log', basex=10)
                ax.set_yscale('log', basey=10)
            ax.set_xlabel(labels[0], fontsize=labelsize)
            ax.set_ylabel(labels[1], fontsize=labelsize)
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            #ax.set_xlim(0,1)
            #ax.set_ylim(0,1)
            
        if n >= 3:
            if colors is None:
                colors = 'k'
            for i, j in np.ndindex((n, n)):
                x = self.iloc[:, i]
                y = self.iloc[:, j]

                ax = fig.add_subplot(n, n, i*n+j+1)

                if nodiag and i == j:
                    x = y = []
                
                ax.scatter(x, y, s=2, marker='o', facecolor=colors, lw=0)

                if pts is not None:
                    for pt, c in zip(pts, colors):
                        ax.scatter([pt[i]],[pt[j]], marker='o', color=c, s=3)  
                if log10:
                    ax.set_xscale('log', basex=10)
                    ax.set_yscale('log', basey=10)

                ax.set_xticks([])
                ax.set_yticks([])
                
                xmin, xmax = ax.get_xlim()  
                ymin, ymax = ax.get_ylim()
                ax.plot([xmin, xmax], [ymin, ymin], lw=2, color='k')
                ax.plot([xmin, xmax], [ymax, ymax], lw=2, color='k')
                ax.plot([xmin, xmin], [ymin, ymax], lw=2, color='k')
                ax.plot([xmax, xmax], [ymin, ymax], lw=2, color='k')

                if i == 0:
                    ax.set_xlabel(labels[j], fontsize=labelsize)
                    ax.xaxis.set_label_position('top')
                if i == n-1:
                    ax.set_xlabel(labels[j], fontsize=labelsize)
                if j == 0:
                    ax.set_ylabel(labels[i], fontsize=labelsize)
                if j == n-1:
                    ax.set_ylabel(labels[i], fontsize=labelsize, rotation=270)
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.labelpad = 20
                
                if lims is not None:
                    ax.set_xlim(lims[j])
                    ax.set_ylim(lims[i])
        
        kwargs = {'wspace':0, 'hspace':0, 'top':0.9, 'bottom':0.1, 
                  'left':0.1, 'right':0.9}    
        if adjust:
            kwargs.update(adjust)
        plt.subplots_adjust(**kwargs)
        plt.savefig(filepath)
        if show:
            plt.show()
        plt.close()


    def hist(self, log10=False, figsize=None, 
             xlims=None, subplots_adjust=None, filepath='', **kwargs_hist):
        """

        :param log10:
        :param figsize:
        :param xlims:
        :param subplots_adjust:
        :param filepath:
        :param kwargs_hist:
        """
        fig = plt.figure(figsize=figsize)

        for i in range(self.ncol):
            ax = fig.add_subplot(self.ncol, 1, i+1)

            if log10:
                dat = np.log10(self.iloc[:,i])
                label = 'log10(%s)'%self.columns[i] 
            else:
                dat = self.iloc[:,i]
                label = self.columns[i]

            if xlims is not None:
                ax.set_xlim(xlims[i])

            ax.hist(dat, **kwargs_hist)
            ax.set_ylabel(label, rotation='horizontal', labelpad=20)
            ax.set_yticklabels([])
            ax.plot(xlims[i], [0,0], '-k', lw=2)
            
            ax.grid(False)
            
            if i != self.ncol-1:
                ax.set_xticklabels([])
            
        kwargs_adjust = {'wspace':0, 'hspace':0, 'top':0.9, 'bottom':0.1, 
                         'left':0.1, 'right':0.9}    
        if subplots_adjust:
            kwargs_adjust.update(subplots_adjust)
        plt.subplots_adjust(**kwargs_adjust)
        
        plt.savefig(filepath)
        plt.show()
        plt.close()



def sampling(func, nstep, p0=None, in_logp=True, seed=None, 
             scheme_sampling='jtj', 
             w1=1, w2=1, temperature=1, stepscale=1, 
             cutoff_singval=0, recalc_sampling_mat=False, 
             interval_print_step=np.inf,
             maxhour=np.inf, filepath='', **kwargs):
    """Perform sampling in parameter space using one of the following two
    schemes:
        - *predict* with a *prior* (eg, Jeffreys prior)
        - *residual* with a *posterior* 
    with tunable weights of prior and data. 

    :param func: predict or residual
    :param p0: 
    :param sampmat: either 'jtj' or 'eye' or a matrix
    :param w1: 
    :param w2: floats between 0 and 1; weights on the prior and data
            components of posterior distribution respectively;
            Pr(p|d) \propto Pr(p) Pr(d|p) \propto Pr(p) exp(-C(p))
            = exp(log(Pr(p)) - C(p)) = exp(w1 log(Pr(p)) - w2 C(p)), where
            w1 and w2 are 1; 
            when w1 goes from 1 to 0, prior gets increasingly flat (just data); 
            when w2 goes from 1 to 0, data gets increasingly uncertain (just prior);
            so, (w1, w2) = (1, 0) means prior sampling,
                (w1, w2) = (0, 1) means data sampling,
                (w1, w2) = (1, 1) means bayesian sampling (default).
            Not implemented yet.
    :param temperature:
    :param stepscale:
    :param cutoff_singval:
    :param recalc_sampling_mat:
    :param interval_print_step:
    :param maxhour:
    :param filepath:
    :param kwargs: a placeholder

    detailed balance:
    Pr(a)Pr(a->b) = Pr(b)Pr(b->a)
    => Pr(a)/Pr(b) = Pr(b->a)/Pr(a->b) 
    = T(b->a)A(b->a)/(T(a->b)A(a->b))

    If using a fixed candidate generating Gaussian density (usually the case),
    T(b->a) = T(a->b), then
    Pr(a)/Pr(b) = A(b->a)/A(a->b)

    Metropolis choice of A(a->b): 
    https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    A(a->b) = min(1, Pr(b)/Pr(a))
    """
    # calculate energies (negative log-prior, cost and their sum)
    eids = ['nlprior', 'cost', 'energy']
    def _get_energies(p):
        try:
            if func.prior is None:
                nlprior = 0
            else:
                nlprior = -np.log(func.prior(p))
            cost = _get_cost(p)
        except:
            print("Error in step %d with parameter:\n%s"%(nstep_tried, str(p)))
            nlprior, cost = np.inf, np.inf
        energy = nlprior + cost
        return nlprior, cost, energy

    # calculate sampling matrix
    def _get_smat(p, scheme, cutoff_singval, stepscale, temperature):
        if scheme == 'jtj':
            jac = Dfunc(p)
            jtj = np.dot(jac.T, jac)
            smat = _hess2smat(jtj, cutoff_singval, stepscale, temperature)
        if scheme == 'eye':
            smat = np.eye(len(p)) * stepscale
        return smat
    
    # initializations
    if p0 is None:
        p0 = func._p0

    if in_logp:
        func = func.get_in_logp()
        p0 = np.log(p0)

    if hasattr(func, 'dat'):  # residual
        Dfunc = func._Dr
        _get_cost = lambda p: func.cost(p)
    else:  # predict
        Dfunc = func._Df
        _get_cost = lambda p: 0

    if seed is None:
        seed = int(time.time() % 1e6)
    np.random.seed(seed)

    nstep_accepted, nstep_tried = 0, 0

    e0 = _get_energies(p0)
    _ens = [(p0, e0)]    

    t0 = time.time()
    smat = _get_smat(p0, scheme_sampling, cutoff_singval, stepscale, 
                     temperature)
    p, e = p0, e0
    
    # start sampling
    while nstep_tried <= nstep and (time.time()-t0) < maxhour*3600:

        if nstep_tried != 0 and nstep_tried % interval_print_step == 0:
            print(nstep_tried)

        if recalc_sampling_mat:
            smat = _get_smat(p, scheme_sampling, cutoff_singval, stepscale, 
                             temperature)
            
        # trial move
        p2 = p + _smat2deltap(smat)
        e2 = _get_energies(p2)
        nstep_tried += 1

        # basic Metropolis accept/reject step                                                                                                                                 
        if recalc_sampling_mat:
            raise NotImplementedError  # the theory is in Gutenkunst's thesis
        else:
            accept = np.random.rand() < np.exp((e[2]-e2[2])/temperature)
        # add p to ens
        if accept:
            _ens.append((p2, e2))
            p, e = p2, e2
            nstep_accepted += 1
        else:
            _ens.append((p, e))

    if in_logp:
        _ens = [(np.exp(p), e) for p, e in _ens]
        pids = map(lambda pid: pid.lstrip('log_'), func.pids)

    columns = pd.MultiIndex.from_tuples([('p', pid) for pid in pids]+
                                        [('e', eid) for eid in eids])
    ens = Ensemble([util.flatten(pe) for pe in _ens], columns=columns)
                   
    if filepath:
        ens.to_csv(filepath)

    return ens



def _hess2smat(hess, cutoff_singval, stepscale, temperature):
    """Convert Hessian to sampling matrix, where sampling matrix is the 
    square root of covariance matrix used for generating Gaussian 
    random vectors.
    
    Hessian is d^2 C / d p_i d p_j, where C = r^T r / 2. 
    
    Geometrically, the anisotropic elliptical parameter cloud is represented
    by Hessian: its eigenvectors correspond to the axis directions, and the 
    reciprocals of square roots of its eigenvalues correpond to the axis lengths. 
    Columns of sampling matrix correspond to the axis directions with 
    appropriate lengths.
    """
    eigvals, eigvecs = np.linalg.eig(hess)
    singvals = np.sqrt(eigvals)
    singval_min = cutoff_singval * max(singvals)
    lengths = 1.0 / np.maximum(singvals, singval_min)

    # now fill in the sampling matrix ("square root" of the Hessian)                                                                                                         
    smat = eigvecs * lengths

    # Divide the sampling matrix by an additional factor such                                                                                                                 
    # that the expected quadratic increase in cost will be about 1.                                                                                                           
    # TODO: the following code block is taken from SloppyCell but what it does
    # seems mysterious
    cutoff_vals = np.compress(singvals < cutoff_singval, singvals)
    if len(cutoff_vals):
        scale = np.sqrt(len(singvals) - len(cutoff_vals) 
                        + sum(cutoff_vals)/cutoff_singval)
    else:
        scale = np.sqrt(len(singvals))
    
    smat /= scale
    smat *= stepscale
    smat *= np.sqrt(temperature)

    return smat



def _smat2deltap(smat):
    """
    """
    randvec = np.random.randn(len(smat))
    deltap = np.dot(smat, randvec)
    return deltap
