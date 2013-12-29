import sys, math, optparse
import numpy as np
import numpy.random as rng
import pylab as pl
import matplotlib.pyplot as plt


##############################################################################

if __name__ == "__main__":
 
    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-n",type = "int",dest = "N",default=6,
                      help="number of subplots on a side")
    parser.add_option("-c",type = "int",dest = "C",default=1000,
                      help="number of samples to draw from each distribution")
    opts, args = parser.parse_args()
    N = opts.N
    C = opts.C
    t = np.arange(0.0, 1.0, 0.1)
    s = np.sin(2*np.pi*t)
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(8,8))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    axisNum = 0
    expansion_factor = 5
    for row in range(N):
        rowval = max(expansion_factor*(N-1-row),1)
        for col in range(N):
            colval = max(expansion_factor*col,1)
            axisNum += 1
            ax = axes[row,col] #plt.subplot(N, N, axisNum)
            color = colors[axisNum % len(colors)]
            color = 'gray'
            S=0
            for c in range(C):
                pvals = np.ravel(rng.dirichlet([colval,rowval],1))
                S = S + rng.multinomial(1,pvals,1)
            vals = S.ravel()

            index = np.arange(2)
            rects = ax.bar(index,vals, orientation='vertical',color=color)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
            ax.text(0,-C/5,'%d,%d'%(colval,rowval), color='b', fontsize=10)
            ax.axis([0,2,0,C])
            if colval>rowval:
                ax.cla()
                ax.axis('off')

    fig.suptitle(r'%d samples each, from DirMults with given $\alpha_0,\alpha_1$'%(C), fontsize=20, fontweight='bold',color='blue')
    #plt.show()
    outfile = 'samples_from_dirichlet_multinomial'
    plt.savefig(outfile)
    print 'Wrote %s.png' % (outfile)


