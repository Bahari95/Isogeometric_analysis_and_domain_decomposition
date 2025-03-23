from simplines import pyccel_sol_field_2d
#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
import numpy                        as     np
#---Plot the solution
'''
TODO : automate the plot of the solution fr any number of patches
'''

def plotddm_result(nbpts, xuh, V, xmp):
    iter_max  = len(xuh[0])
    numPaches = len(V) 
    #---Compute a solution
    F1 = []
    for i in range(numPaches):
        F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
        #F2   = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
    u = []
    for ii in range(numPaches):
        u1    = []
        for i in range(iter_max):
            u1.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh[ii][i], V[ii].knots, V[ii].degree)[0][:,50])
        u.append(u1)
    plt.figure() 
    plt.axes().set_aspect('equal')
    plt.subplot(121)
    for ii in range(numPaches):
        for i in range(iter_max-1):
            plt.plot(F1[ii][:,50], u[ii][i], '-k', linewidth = 1.)
    for ii in range(numPaches):
        plt.plot(F1[ii][:,50], u[ii][-1], '-k', linewidth = 1., label='u_{}'.format(ii))
    plt.legend()
    plt.grid(True)  
    plt.subplot(122)
    plt.plot(F1[ii][:,50], u[ii][-1],  '--or', label = '$Un_{}-iter-max'.format(ii))
    plt.legend()
    plt.grid(True)  
    plt.savefig('behvoir_between_two_patches.png')
    return 0