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
    
    
    
def plot_Mesh(nbpts, V, xmp, ymp, cp = True, savefig = None, plot = True): 
   ''''
   Plot the solution of the problem in the whole domain
   '''
   #---Compute a solution
  
   
  
   F1x = pyccel_sol_field_2d((nbpts, nbpts), xmp, V.knots, V.degree)[0]
   F1y = pyccel_sol_field_2d((nbpts, nbpts), ymp, V.knots, V.degree)[0]
      

   # --- Create Figure ---
   fig =plt.figure() 
   for i in range(nbpts):
         phidx = F1x[:,i]
         phidy = F1y[:,i]
         plt.plot(phidx, phidy, linewidth = 0.5, color = 'b')
   for i in range(nbpts):
         phidx = F1x[i,:]
         phidy = F1y[i,:]

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'b')
   phidx = F1x[:,0]
   phidy = F1y[:,0]
   plt.plot(phidx, phidy, '-k', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
   # ...
   phidx = F1x[:,nbpts-1]
   phidy = F1y[:,nbpts-1]
   plt.plot(phidx, phidy, '-g', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
   
   
   
   phidx = F1x[0,:]
   phidy = F1y[0,:]
   plt.plot(phidx, phidy, '-r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
   # ...
   phidx = F1x[nbpts-1,:]
   phidy = F1y[nbpts-1,:]
   plt.plot(phidx, phidy, '-r', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0
   
   



def plot_surface_solution(nbpts, xuh, V, xmp, ymp, title="Surface Plot",
                          savefig=None, plot=True, Jacfield=None, U_ex=None):
    '''
    Plot the numerical solution as a 3D surface in the whole domain,
    and optionally compare it with the exact solution U_ex(x, y).
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: needed for 3D plotting
    import numpy as np

    # Compute numerical solution and physical coordinates
    un, uxn, uyn, X, Y = pyccel_sol_field_2d((nbpts, nbpts), xuh, V.knots, V.degree)
    Fx = pyccel_sol_field_2d((nbpts, nbpts), xmp, V.knots, V.degree)[0]
    Fy = pyccel_sol_field_2d((nbpts, nbpts), ymp, V.knots, V.degree)[0]
    Fx_x, Fx_y = pyccel_sol_field_2d((nbpts, nbpts), xmp, V.knots, V.degree)[1:3]
    Fy_x, Fy_y = pyccel_sol_field_2d((nbpts, nbpts), ymp, V.knots, V.degree)[1:3]

    # Jacobian determinant override (if provided)
    if Jacfield is not None:
        JF = Fx_x * Fy_y - Fx_y * Fy_x
        un = JF

    # Evaluate exact solution if provided
    if U_ex is not None:
        U_exact = U_ex(Fx, Fy)       

        fig = plt.figure(figsize=(12, 4), dpi=150)

        # Plot numerical solution
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(Fx, Fy, un, cmap='viridis', edgecolor='none')
        ax1.set_title(title + " (Numerical)", fontsize=10)
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("u_h")
        fig.colorbar(surf1, ax=ax1, shrink=0.5)

        # Plot exact solution
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(Fx, Fy, U_exact, cmap='plasma', edgecolor='none')
        ax2.set_title("Exact Solution", fontsize=10)
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("u_ex")
        fig.colorbar(surf2, ax=ax2, shrink=0.5)

        

        fig.tight_layout()

    else:
        # Plot only numerical solution
        fig = plt.figure(figsize=(7, 4), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Fx, Fy, un, cmap='jet', edgecolor='none')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
        fig.colorbar(surf, shrink=0.5)
        fig.tight_layout()

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show(block=plot)

    print("3D Surface plot done.")
    return 0


def plot_Solution(nbpts, xuh, V, xmp, ymp, savefig=None, plot=True, Jacfield=None):
    '''
    Plot the solution of the problem in the whole domain
    '''
    # Get solution and coordinates
    u = pyccel_sol_field_2d((nbpts, nbpts), xuh, V.knots, V.degree)[0]
    F1 = pyccel_sol_field_2d((nbpts, nbpts), xmp, V.knots, V.degree)[0]
    F2 = pyccel_sol_field_2d((nbpts, nbpts), ymp, V.knots, V.degree)[0]
    F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp, V.knots, V.degree)[1:3]
    F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp, V.knots, V.degree)[1:3]

    # If needed, replace u with Jacobian field
    if Jacfield is not None:
        JF = F1x * F2y - F1y * F2x
        u = JF

    # Levels for contour
    u_min = np.min(u)
    u_max = np.max(u)
    levels = np.linspace(u_min, u_max + 1e-10, 100)

    # Create plot
    fig, axes = plt.subplots(figsize=(8, 6))
    im = axes.contourf(F1, F2, u, levels=levels, cmap='jet')

    # Colorbar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.label.set_fontweight('bold')

    # Formatting
    axes.set_title("Solution in the whole domain", fontweight='bold')
    axes.set_xlabel("x", fontweight='bold')
    axes.set_ylabel("y", fontweight='bold')
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontweight('bold')

    fig.tight_layout()

    # Save or show
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show(block=plot)

    print("Plotting done: Solution in the whole domain.")
    return 0


