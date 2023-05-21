from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

# .. Matrices in 1D ..
from gallery_section_04 import assemble_stiffnessmatrix1D
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
#---In Poisson equation
from gallery_section_04 import assemble_vector_ex01   
from gallery_section_04 import assemble_norm_ex01     

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

   def __init__(self, V, V2, S_DDM, domain_nb, ovlp_value):
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       K                  = assemble_stiffness1D(V)
       K                  = K.tosparse()
       if domain_nb == 0 :
          K               = K.toarray()[1:,1:]
          K[-1,-1]       += S_DDM
       else :
          K               = K.toarray()[:-1,:-1]
          K[0,0]         += S_DDM
       K                  = csr_matrix(K)

       # ...
       self.lu             = sla.splu(csc_matrix(K))
       
       # ++++
       self.spaces        = [V, V2]
       self.ovlp_value    = ovlp_value
       self.S_DDM         = S_DDM
       self.domain_nb     = domain_nb
   def solve(self, u_d):

       V, Vh               = self.spaces[:]

       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs( Vh, fields = [u_d], knots = True, value = [self.ovlp_value, self.S_DDM, self.domain_nb], out = rhs )
       b                   = rhs.toarray()
       if self.domain_nb == 0 :
          b                   = b[1:]      
       else :
          b                   = b[:-1] 

       xkron               = self.lu.solve(b)       
       # ...
       x                   = np.zeros(V.nbasis)
       if self.domain_nb == 0 :
          x[1:]  = xkron[:]
       else :
          x[:-1] = xkron[:]
       #...
       u  = StencilVector(V.vector_space)
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1
nelements   = 1024

# ... please take into account that : beta < alpha 
alpha       = 0.5
beta        = 0.5
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 10
S_DDM       = 1./(beta)
#--------------------------
#..... Initialisation
#--------------------------
grids_0 = linspace(0, alpha, nelements+1)
# create the spline space for each direction
V_0    = SplineSpace(degree=degree, nelements= nelements, grid =grids_0, nderiv = 2, quad_degree = quad_degree)

grids_1 = linspace(beta, 1., nelements+1)
# create the spline space for each direction
V_1    = SplineSpace(degree=degree, nelements= nelements, grid =grids_1, nderiv = 2, quad_degree = quad_degree)

Vt_0    = TensorSpace(V_0, V_1)
Vt_1    = TensorSpace(V_1, V_0)

DDM_0 = DDM_poisson( V_0, Vt_0,  S_DDM, 0, alpha )
DDM_1 = DDM_poisson( V_1, Vt_1,  S_DDM, 1, beta )

# ... communication Dirichlet interface
u_00    = StencilVector(V_0.vector_space)
u_1     = StencilVector(V_1.vector_space)

print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve( u_1)
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1   = DDM_1.solve( u_00)
xuh_01.append(xuh_1)
u_00 = u_0
l2_err = l2_norm + l2_norm1
H1_err = H1_norm + H1_norm1
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

for i in range(iter_max):
	#...
	u_0, xuh, l2_norm, H1_norm     = DDM_0.solve(u_1)
	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1 = DDM_1.solve(u_00)
	xuh_01.append(xuh_1)
	u_00   = u_0
	l2_err = l2_norm + l2_norm1
	H1_err = H1_norm + H1_norm1
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

#---Compute a solution
from simplines import plot_field_1d
nbpts = 100
plt.figure()
plot_field_1d(V_0.knots, V_0.degree, xuh, nx=101, color='b')
plot_field_1d(V_1.knots, V_1.degree, xuh_1, nx=101, color='r')
plt.show()

# # ........................................................
# ....................For a plot
# #.........................................................
if True :
	plt.figure()
	for i in range(iter_max):
		plot_field_1d(V_0.knots, V_0.degree, xuh_0[i],  nx=101, color='b')
		plot_field_1d(V_1.knots, V_1.degree, xuh_01[i], nx=101, color='r')
	plt.show()
