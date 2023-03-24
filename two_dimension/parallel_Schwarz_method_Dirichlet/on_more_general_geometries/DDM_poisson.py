from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

# .. Matrices in 1D ..
from gallery_section_04 import assemble_stiffnessmatrix1D
from gallery_section_04 import assemble_massmatrix1D
from gallery_section_04 import assemble_matrix_ex11
from gallery_section_04 import assemble_matrix_ex12
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex11, arity=2)
assemble_matrix_ex10 = compile_kernel(assemble_matrix_ex12, arity=2)

#---In Poisson equation
from gallery_section_04 import assemble_vector_ex01   
from gallery_section_04 import assemble_matrix_un_ex01
from gallery_section_04 import assemble_norm_ex01      
from gallery_section_04 import assemble_vector_ex02   
from gallery_section_04 import assemble_matrix_ex01   


assemble_mass2D      = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_stiffness3D = compile_kernel(assemble_matrix_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)
assemble_Pr          = compile_kernel(assemble_vector_ex02, arity=1)

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray, linspace
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   tabulate                     import tabulate
import numpy                        as     np
import timeit
import time

from tabulate import tabulate

#============================================================================== 
# ... projection of a solution in a sub-domain to another sub-domain
#============================================================================== 
def   Pr_h_solve(V1, V2, V, Vt, u, domain_nb, ovlp_value): 

       # ...
       stiffness     = assemble_mass2D(V1, value = [ovlp_value])
       M             = stiffness.tosparse()
       lu            = sla.splu(csc_matrix(M))
       # ...
       rhs           = StencilVector(V1.vector_space)
       rhs           = assemble_Pr(Vt, fields = [u], knots = True, value = [ovlp_value], out = rhs) 
       b             = rhs.toarray()
       
       #---Solve a linear system
       x             = lu.solve(b)
       # ---
       x_n           = zeros(V.nbasis)
       if domain_nb == 0 :
          x_n[-1,:]  = x[:]
       else :
          x_n[0,:]   = x[:]       
       # ---
       u_L2          = StencilVector(V.vector_space)
       u_L2.from_array(V, x_n)
       
       return u_L2, x_n
       
#==============================================================================
#.......Poisson ALGORITHM
#==============================================================================
def poisson_solve(V1, V2, V, u_d = None):
       u                   = StencilVector(V.vector_space)
       # ++++
       #..Stiffness matrix in 3D
       stiffness           = assemble_stiffness3D(V)
       stiffness           = apply_dirichlet(V, stiffness)
       # ...
       M                   = stiffness.tosparse()
       lu                  = sla.splu(csc_matrix(M))
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V, fields = [u_d] )
       rhs                 = apply_dirichlet(V, rhs)
       b                   = rhs.toarray()
       # ...
       x                   = lu.solve(b)       
       x                   = x.reshape(V.nbasis)
       # ...
       #... Dirichlet nboundary
       x[0,:]              = u_d.toarray().reshape(V.nbasis)[0,:]
       x[-1,:]             = u_d.toarray().reshape(V.nbasis)[-1,:]
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1

# ... please take into account that : beta < alpha 
alpha       = 0.65
beta        = 0.35
overlap     = alpha - beta
xuh_0    = []
xuh_01   = []
iter_max = 20

#------------------------------------------------------------
#..... Initialisation 
#------------------------------------------------------------
nelements  = 32

grids_0 = linspace(0, alpha, nelements+1)
# create the spline space for each direction
V1_0    = SplineSpace(degree=degree, nelements= nelements, grid =grids_0, nderiv = 2, quad_degree = quad_degree)
V2_0    = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V_0     = TensorSpace(V1_0, V2_0)

grids_1 = linspace(beta, 1., nelements+1)
# create the spline space for each direction
V1_1    = SplineSpace(degree=degree, nelements= nelements, grid =grids_1, nderiv = 2, quad_degree = quad_degree)
V2_1    = SplineSpace(degree=degree, nelements= nelements,  nderiv = 2, quad_degree = quad_degree)
V_1     = TensorSpace(V1_1, V2_1)
#...
Vt_0    = TensorSpace(V2_0, V1_1, V2_1)
Vt_1    = TensorSpace(V2_1, V1_0, V2_0)
# ... communication Dirichlet interface
uh_d1   = StencilVector(V_0.vector_space)
uh_d0   = StencilVector(V_1.vector_space)

print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = poisson_solve(V1_0, V2_0, V_0, u_d= uh_d1)
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1   = poisson_solve(V1_1, V2_1, V_1, u_d= uh_d0)
xuh_01.append(xuh_1)
l2_err = l2_norm + l2_norm1
H1_err = H1_norm + H1_norm1
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

for i in range(iter_max):
	# ... Dirichlezt boudndary condition in x = 0.75 and 0.25
	uh_d1, xh_d       = Pr_h_solve(V2_0, V2_1, V_0, Vt_0, u_1, 0, alpha)
	uh_d0, xh         = Pr_h_solve(V2_1, V2_0, V_1, Vt_1, u_0, 1, beta)
	#...
	u_0,   xuh, l2_norm, H1_norm     = poisson_solve(V1_0, V2_0, V_0, u_d= uh_d1)
	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1   = poisson_solve(V1_1, V2_1, V_1, u_d= uh_d0)
	xuh_01.append(xuh_1)
	l2_err = l2_norm + l2_norm1
	H1_err = H1_norm + H1_norm1
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

#---Compute a solution
nbpts = 100
# # ........................................................
# ....................For a plot
# #.........................................................
if True :
	#---Compute a solution
	u, ux, uy, sX, sY               = pyccel_sol_field_2d((nbpts, nbpts),  xuh,   V_0.knots, V_0.degree)
	# ...
	X = (2.0*sX-1.0) * sqrt(1.-0.5*(2.0*sY-1.0)**2)
	Y = (2.0*sY-1.0) * sqrt(1.-0.5*(2.0*sX-1.0)**2)
	# ...
	u_1, ux_1, uy_1, sX_1, sY_1     = pyccel_sol_field_2d((nbpts, nbpts),  xuh_1,   V_1.knots, V_1.degree)
	# ...
	X_1 = (2.0*sX_1-1.0) * sqrt(1.-0.5*(2.0*sY_1-1.0)**2)
	Y_1 = (2.0*sY_1-1.0) * sqrt(1.-0.5*(2.0*sX_1-1.0)**2)
	u_0  = []
	u_01 = []
	for i in range(iter_max):
	    u_0.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_0[i],   V_0.knots, V_0.degree)[0][:,50])
	    u_01.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_01[i],   V_1.knots, V_1.degree)[0][:,50])
	# ...
	#solut = lambda  x, y : exp(-500*(x**2 + y**2 - 0.2)**2)
	solut = lambda  x, y : 1.-x**2-y**2

	plt.figure() 
	plt.axes().set_aspect('equal')
	plt.subplot(121)
	for i in range(iter_max-1):
	     plt.plot(X[:,50], u_0[i], '-k', linewidth = 1.)
	     plt.plot(X_1[:,50], u_01[i], '-k', linewidth = 1.)
	plt.plot(X_1[:,50], u_01[i+1], '-k', linewidth = 1., label='$\mathbf{Un_1-iter(i)}$')
	plt.plot(X[:,50], u_0[i+1], '-k', linewidth = 1., label='$\mathbf{Un_0-iter(i)}$')
	plt.grid(True)
	plt.legend()
	plt.subplot(122)
	plt.plot(X[:,50], u[:,50],  '--or', label = '$\mathbf{Un_0-iter-max}$' )
	plt.plot(X_1[:,50], u_1[:,50],  '--om', label = '$\mathbf{Un_1-iter-max}$')
	plt.grid(True)
	plt.legend()
	#plt.savefig('figs/Pu_{}.png'.format(0))
	plt.show()
	# set up a figure twice as wide as it is tall
	fig = plt.figure(figsize=plt.figaspect(0.5))
	#===============
	# First subplot
	# set up the axes for the first plot
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	# plot a 3D surface like in the example mplot3d/surface3d_demo
	surf0 = ax.plot_surface(X[:,:], Y[:,:], u[:,:], rstride=1, cstride=1, cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
	surf0 = ax.plot_surface(X_1[:,:], Y_1[:,:], u_1[:,:], rstride=1, cstride=1, cmap='viridis',
		               linewidth=0, antialiased=False)
	ax.set_xlim(-1.0, 1.0)
	ax.set_ylim(-1.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate solution in uniform mesh')
	ax.set_xlabel('X',  fontweight ='bold')
	ax.set_ylabel('Y',  fontweight ='bold')
	# Add a color bar which maps values to colors.
	fig.colorbar(surf0, shrink=0.5, aspect=25)

	#===============
	# Second subplot
	ax = fig.add_subplot(1, 2, 2, projection='3d')
	surf = ax.plot_surface(X_1[:,:], Y_1[:,:], solut(X_1[:,:], Y_1[:,:]), cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
	surf = ax.plot_surface(X[:,:], Y[:,:], solut(X[:,:], Y[:,:]), cmap='viridis',
		               linewidth=0, antialiased=False)
	ax.set_xlim(-1.0, 1.0)
	ax.set_ylim(-1.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate Solution in adaptive meshes')
	ax.set_xlabel('F1',  fontweight ='bold')
	ax.set_ylabel('F2',  fontweight ='bold')
	fig.colorbar(surf, shrink=0.5, aspect=25)
	plt.savefig('Poisson3D.png')
	plt.show()
