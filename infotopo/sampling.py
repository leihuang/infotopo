"""
"""

try:
    from __future__ import division
except:
    pass

import pandas as pd



class Ensemble(pd.DataFrame):
    def __init__(self):
        pass

    def scatter(self, hist=False, log10=False, pts=None, colors=None,
                figsize=None, adjust=None, labels=None, labelsize=6,
                nodiag=False, lims=None,
                filepath='', show=False):
        """
        :param hist: if True, also plot histograms for the marginal distributions
        :param filepath:
        """
        n = self.ncol
        assert n > 1, "Cannot do scatterplot with 1d data."
        
        if figsize is None:
            figsize = (n*2, n*2)
        fig = plt.figure(figsize=figsize)

        if n == 2:
            ax = fig.add_subplot(111)
            xs, ys = self.iloc[:,0], self.iloc[:,1]
            ax.scatter(xs, ys, s=1)
            if pts is not None:
                for pt in pts:
                    # can change the color for diff pts
                    ax.scatter(*pt, marker='o', color='r', s=10)  
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())            
            if log10:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel(self.colvarids[0], fontsize=10)
            ax.set_ylabel(self.colvarids[1], fontsize=10)
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            
        if n >= 3:
            if colors is None:
                colors = 'k'
            if labels is None:
                labels = self.colvarids
            for i, j in np.ndindex((n, n)):
                ens_i = self.iloc[:, i]
                ens_j = self.iloc[:, j]
                varid_i = labels[i]
                varid_j = labels[j]
                ax = fig.add_subplot(n, n, i*n+j+1)
                if nodiag:
                    if i == j:
                        ens_i = []
                        ens_j = []
                ax.scatter(ens_j, ens_i, s=2, marker='o', facecolor=colors, 
                           lw=0)
                if pts is not None:
                    for pt in pts:
                        # can change the color for diff pts
                        ax.scatter([pt[i]],[pt[j]], marker='o', color='r', s=3)  
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
                    ax.set_xlabel(varid_j, fontsize=labelsize)
                    ax.xaxis.set_label_position('top')
                if i == n-1:
                    ax.set_xlabel(varid_j, fontsize=labelsize)
                if j == 0:
                    ax.set_ylabel(varid_i, fontsize=labelsize)
                if j == n-1:
                    ax.set_ylabel(varid_i, fontsize=labelsize, rotation=270)
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



def sampling():
    pass