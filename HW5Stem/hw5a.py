#edit
# region imports
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# endregion

# region functions
def ff(Re, rr, CBEQN=False):
    """
    This function calculates the friction factor for a pipe based on the
    notion of laminar, turbulent and transitional flow.
    :param Re: the Reynolds number under question.
    :param rr: the relative pipe roughness (expect between 0 and 0.05)
    :param CBEQN:  boolean to indicate if I should use Colebrook (True) or laminar equation
    :return: the (Darcy) friction factor
    """
    if(CBEQN):
        # note:  in numpy log is for natural log.  log10 is log base 10.
        cb=lambda f: #JES MISSING CODE
        result= #JES MISSING CODE  # use fsolve to find the frction factor using Colebrook equation
        return result[0]
    else:
        return 64/Re
    pass

def plotMoody(plotPoint=False, pt=(0,0)):
    """
    This function produces the Moody diagram for a Re range from 1 to 10^8 and
    for relative roughness from 0 to 0.05 (20 steps).  The laminar region is described
    by the simple relationship of f=64/Re whereas the turbulent region is described by
    the Colebrook equation.
    :return: just shows the plot, nothing returned
    """
    #Step 1:  create logspace arrays for ranges of Re
    ReValsCB=np.logspace(#JES MISSING CODE)  # for use with Colebrook equation (i.e., Re in range from 4000 to 10^8)
    ReValsL=np.logspace(np.log10(600.0),np.log10(2000.0),20)  # for use with Laminar flow (i.e., Re in range from 600 to 2000)
    ReValsTrans=np.logspace(#JES MISSING CODE)  # for use with Transition flow (i.e., Re in range from 2000 to 4000)
    #Step 2:  create array for range of relative roughnesses
    rrVals=np.array([0,1E-6,5E-6,1E-5,5E-5,1E-4,2E-4,4E-4,6E-4,8E-4,1E-3,2E-3,4E-3,6E-3,8E-8,1.5E-2,2E-2,3E-2,4E-2,5E-2])

    #Step 2:  calculate the friction factor in the laminar range
    ffLam=np.array([#JES MISSING CODE]) # use list comprehension for all Re in ReValsL and calling ff
    ffTrans=np.array([#JES MISSING CODE]) # use list comprehension for all Re in ReValsTrans and calling ff

    #Step 3:  calculate friction factor values for each rr at each Re for turbulent range.
    ffCB=np.array([[#JES MISSING CODE] for relRough in rrVals])

    #Step 4:  construct the plot
    plt.loglog(#JES MISSING CODE)  # plot the laminar part as a solid line
    plt.loglog(#JES MISSING CODE)  # plot the transition part as a dashed line
    for nRelR in range(len(ffCB)):
        plt.loglog(#JES MISSING CODE, color='k', label=nRelR)  # plot the lines for the turbulent region for each pipe roughness
        plt.annotate(xy=(#JES MISSING CODE),text=rrVals[nRelR])  # put a label at end of each curve on the right

    plt.xlim(600,1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $#JES MISSING CODE$", fontsize=16)
    plt.ylabel(r"Friction factor $#JES MISSING CODE$", fontsize=16)
    plt.text(2.5E8,0.02,r"Relative roughness $\frac{\epsilon}{d}$",rotation=90, fontsize=16)
    ax = plt.gca()  # capture the current axes for use in modifying ticks, grids, etc.
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)  # format tick marks
    ax.tick_params(axis='both', grid_linewidth=1, grid_linestyle='solid', grid_alpha=0.5)
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both')
    if(plotPoint):
        plt.plot(#JES MISSING CODE, markersize=12, markeredgecolor='red', markerfacecolor='none')

    plt.show()

def main():
    plotMoody()
# endregion

# region function calls
if __name__=="__main__":
    main()
# endregion