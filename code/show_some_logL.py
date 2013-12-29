import sys, math, optparse
import numpy as np
import numpy.random as rng
import pylab as pl
import matplotlib.pyplot as plt

def calc_lgamma_vect(vect):
    """Calculate the log gamma of each number in a vector """
    v = np.array(vect) #+ 1e-10
    if np.any(v<0.5): print 'FOUND ELEMENT <0.5 \n %s \n' % v
    for i in range(v.size):       
        v[i] = math.lgamma(v[i])
    return v

def calc_DirMult_logL(n, alphas):
    """ Calculate the log likelihood under DirMult distribution with
    alphas, given data counts of n"""
    #1st two terms for full calculation
    sum_alphas = np.sum(alphas)
    lg_sum_alphas = math.lgamma(sum_alphas)
    sum_lg_alphas = np.sum(calc_lgamma_vect(alphas))

    lg_sum_alphas_n = math.lgamma(sum_alphas + np.sum(n))
    sum_lg_alphas_n = np.sum(calc_lgamma_vect(n+alphas))

    logL = lg_sum_alphas - sum_lg_alphas - lg_sum_alphas_n + sum_lg_alphas_n
    return logL

def make_array_of_logLs(counts, rowvals, colvals):
    array_of_logLs = np.zeros((len(rowvals),len(colvals)), dtype = float)
    rows, cols = range(len(rowvals)),range(len(colvals))
    for i,rowval in enumerate(rowvals):
        for j,colval in enumerate(colvals):
            alphas = np.array([colval,rowval])
            array_of_logLs[i,j] = calc_DirMult_logL(counts, alphas)
    return array_of_logLs

##############################################################################

if __name__ == "__main__":
 
    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-n","--maxalpha",type = "int",dest = "N",default=10,
                      help="alphas go from 1 up to this number")
    opts, args = parser.parse_args()
    N = opts.N

    fig2 = plt.figure(figsize=(8,8))
    fig2.suptitle(r'logL(counts) as function of $\alpha=(\alpha_0,\alpha_1)$', fontsize=20, fontweight='bold',color='blue')
    rowvals = 1.0 + np.arange(N)
    colvals = 1.0 + np.arange(N)
    print('rowvals: ', rowvals)
    colormap = 'gist_heat' #'RdYlBu'
    #------------------------------------------------------------
    counts = np.array([1,1])
    ax1 = plt.subplot(221)
    A = make_array_of_logLs(counts, rowvals, colvals)
    img = ax1.imshow(A,interpolation='nearest',origin='lowerleft',cmap=colormap)
    plt.colorbar(img, ax=ax1, orientation='vertical',shrink=0.75)
    #ax1.set_xticklabels([1]+colvals.tolist()) # no idea why have to prepend!!
    #ax1.set_yticklabels([1]+rowvals.tolist())
    ax1.set_title('counts of %s'%(str(counts.tolist())))
    ax1.set_xlabel(r'$\alpha_0-1$',fontsize=18)
    ax1.set_ylabel(r'$\alpha_1-1$',fontsize=18)
    #------------------------------------------------------------
    counts = np.array([100,100])
    ax1 = plt.subplot(222)
    A = make_array_of_logLs(counts, rowvals, colvals)
    img = ax1.imshow(A,interpolation='nearest',origin='lowerleft',cmap=colormap)
    plt.colorbar(img, ax=ax1, orientation='vertical',shrink=0.75)
    ax1.set_title('counts of %s'%(str(counts.tolist())))
    #------------------------------------------------------------
    counts = np.array([1,4])
    ax1 = plt.subplot(223)
    A = make_array_of_logLs(counts, rowvals, colvals)
    img = ax1.imshow(A,interpolation='nearest',origin='lowerleft',cmap=colormap)
    plt.colorbar(img, ax=ax1, orientation='vertical',shrink=0.75)
    ax1.set_title('counts of %s'%(str(counts.tolist())))
    #------------------------------------------------------------
    counts = np.array([100,400])
    ax1 = plt.subplot(224)
    A = make_array_of_logLs(counts, rowvals, colvals)
    img = ax1.imshow(A,interpolation='nearest',origin='lowerleft',cmap=colormap)
    plt.colorbar(img, ax=ax1, orientation='vertical',shrink=0.75)
    ax1.set_title('counts of %s'%(str(counts.tolist())))
    #------------------------------------------------------------


    outfile = 'dirichlet_multinomial_logL'
    plt.savefig(outfile)
    print 'Wrote %s.png' % (outfile)
