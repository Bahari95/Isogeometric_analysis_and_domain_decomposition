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

def plot_SolutionMultipatch(nbpts, xuh, V, xmp, ymp): 
    #---Compute a solution
    numPaches = len(V)
    u = []
    F1 = []
    F2 = []
    for i in range(numPaches):
        u.append(pyccel_sol_field_2d((nbpts, nbpts), xuh[i], V[i].knots, V[i].degree)[0])
        #---Compute a solution
        F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
        F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])

    # --- Compute Global Color Levels ---
    u_min  = min(np.min(u[0]), np.min(u[1]))
    u_max  = max(np.max(u[0]), np.max(u[1]))
    for i in range(2, numPaches):
        u_min  = min(u_min, np.min(u[i]))
        u_max  = max(u_max, np.max(u[i]))
    levels = np.linspace(u_min, u_max, 100)  # Uniform levels for both plots

    # --- Create Figure ---
    fig, axes = plt.subplots(figsize=(8, 6))

    # --- Contour Plot for First Subdomain ---
    im = []
    for i in range(numPaches):
        im.append(axes.contourf(F1[i], F2[i], u[i], levels, cmap='jet'))
        # --- Colorbar ---
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
        cbar = plt.colorbar(im[i], cax=cax)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.label.set_fontweight('bold')
    # --- Formatting ---
    axes.set_title("Solution the in whole domain ", fontweight='bold')
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontweight('bold')

    fig.tight_layout()
    plt.savefig('2patch.png')
    plt.show()
    return 0