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

assemble_stiffness2D = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

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

#==============================================================================
#.......Poisson ALGORITHM
class DDM_poisson(object):
   def __init__(self, V1, V2, V, V_TOT, S_DDM, domain_nb, ovlp_value):
       
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       K1                  = assemble_stiffness1D(V1)
       K1                  = K1.tosparse()
       if domain_nb == 0 :
          K1               = K1.toarray()[1:,1:]
       else :
          K1               = K1.toarray()[:-1,:-1]
       K1                  = csr_matrix(K1)

       M1                  = assemble_mass1D(V1)
       M1                  = M1.tosparse()
       if domain_nb == 0 :
           M1                  = M1.toarray()[1:,1:]
       else :
           M1                  = M1.toarray()[:-1,:-1]
       M1                  = csr_matrix(M1)

       # Stiffness and Mass matrix in 1D in the second deriction
       K2                  = assemble_stiffness1D(V2)
       K2                  = K2.tosparse()
       K2                  = K2.toarray()[1:-1,1:-1]
       K2                  = csr_matrix(K2)

       M2                  = assemble_mass1D(V2)
       M2                  = M2.tosparse()
       M2                  = M2.toarray()[1:-1,1:-1]
       M2                  = csr_matrix(M2)
       
       # ...
       Id                  = zeros((V1.nbasis-1,V1.nbasis-1))
       # ... TO integrate the boundary
       if domain_nb == 0 :
           Id[-1,-1]       = S_DDM
       else :
           Id[0,0]         = S_DDM
       # ...
       M                   = kron(K1,M2)+kron(M1,K2) + kron(Id,M2)
       self.lu             = sla.splu(csc_matrix(M))
       self.domain_nb      = domain_nb
       self.S_DDM          = S_DDM
       self.nbasis_1       = (V1.nbasis-1)*(V2.nbasis-2)
       self.nbasis_2       = [V1.nbasis-1,V2.nbasis-2]
       self.ovlp_value     = ovlp_value
       self.sp              = [V, V_TOT]
   def solve(self, u_d):
       
       V, V_TOT            = self.sp[:]
       # ...
       
       u                   = StencilVector(V.vector_space)
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs( V_TOT, fields = [u_d], knots = True, value = [self.ovlp_value, self.S_DDM, self.domain_nb], out = rhs )
       b                   = rhs.toarray()
       b                   = b.reshape(V.nbasis)
       if self.domain_nb == 0 :
          b                   = b[1:, 1:-1]      
       else :
          b                   = b[:-1, 1:-1]      
       b                   = b.reshape(self.nbasis_1)
       # ...
       xkron               = self.lu.solve(b)       
       xkron               = xkron.reshape(self.nbasis_2)
       # ...
       x                   = np.zeros(V.nbasis)
       if self.domain_nb == 0 :
          x[1:, 1:-1]  = xkron
       else :
          x[:-1, 1:-1] = xkron
       #...
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1
nelements   = 32

# ... please take into account that : beta < alpha 
alpha       = 0.5
beta        = 0.5
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 20
S_DDM       = 1./beta
#--------------------------
#..... Initialisation
#--------------------------
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
Vt_0    = TensorSpace(V1_0, V2_0, V1_1, V2_1)
Vt_1    = TensorSpace(V1_1, V2_1, V1_0, V2_0)

# --- Initialization
P0      = DDM_poisson(V1_0, V2_0, V_0, Vt_0, S_DDM, 0, alpha )
P1      = DDM_poisson(V1_1, V2_1, V_1, Vt_1, S_DDM, 1, beta  )

# ... communication Dirichlet interface
u_00    = StencilVector(V_0.vector_space)
u_1     = StencilVector(V_1.vector_space)

print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = P0.solve(u_1)
u_00 = u_0
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1   = P1.solve(u_0)
xuh_01.append(xuh_1)
l2_err = l2_norm + l2_norm1
H1_err = H1_norm + H1_norm1
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

for i in range(iter_max):
	#...
	u_0, xuh, l2_norm, H1_norm     = P0.solve(u_1)
	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1 = P1.solve(u_00)
	xuh_01.append(xuh_1)
	u_00   = u_0
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
	u, ux, uy, X, Y               = pyccel_sol_field_2d((nbpts, nbpts),  xuh,   V_0.knots, V_0.degree)
	# ...
	u_1, ux_1, uy_1, X_1, Y_1     = pyccel_sol_field_2d((nbpts, nbpts),  xuh_1,   V_1.knots, V_1.degree)
	u_0  = []
	u_01 = []
	for i in range(iter_max):
	    u_0.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_0[i],   V_0.knots, V_0.degree)[0][:,50])
	    u_01.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_01[i],   V_1.knots, V_1.degree)[0][:,50])
	# ...
	#solut = lambda x, t, y : sin(pi*t)*x**2*y*3*sin(4.*pi*(1.-x))*(1.-y) 
	solut = lambda  x, y :  sin( pi*x)* sin( pi*y)

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
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
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
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate Solution in adaptive meshes')
	ax.set_xlabel('F1',  fontweight ='bold')
	ax.set_ylabel('F2',  fontweight ='bold')
	fig.colorbar(surf, shrink=0.5, aspect=25)
	plt.savefig('Poisson3D.png')
	plt.show()
