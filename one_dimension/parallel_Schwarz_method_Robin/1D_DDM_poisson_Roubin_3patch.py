from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

# .. Matrices in 1D ..
from gallery_section_04_multipatch import assemble_stiffnessmatrix1D
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
#---In Poisson equation
from gallery_section_04_multipatch import assemble_vector_ex01   
from gallery_section_04_multipatch import assemble_norm_ex01     

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
from simplines import plot_field_1d
#==============================================================================
#.......Poisson ALGORITHM
class DDM_poisson(object):

   def __init__(self, V, V2, S_DDM, domain_nb, ovlp_value_left, ovlp_value_right):
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       K                  = assemble_stiffness1D(V)
       K                  = K.tosparse()
       if domain_nb == 0 :
          K               = K.toarray()[1:,1:]
          K[-1,-1]       += S_DDM
       elif domain_nb == 1 :
          K               = K.toarray()[:-1,:-1]
          K[0,0]         += S_DDM
       else:
       	  K              = K.toarray()
          K[0,0]         += S_DDM
          K[-1,-1]      += S_DDM
          	        
       K                  = csr_matrix(K)

       # ...
       self.lu             = sla.splu(csc_matrix(K))
       
       # ++++
       self.spaces        = [V, V2]
       self.ovlp_value_left  = ovlp_value_left
       self.ovlp_value_right  = ovlp_value_right
       self.S_DDM         = S_DDM
       self.domain_nb     = domain_nb
   def solve(self, u_d1, u_d2):

       V, Vh               = self.spaces[:]

       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs( Vh, fields = [u_d1, u_d2], knots = True, value = [self.ovlp_value_left, self.ovlp_value_right, self.S_DDM, self.domain_nb], out = rhs )
       b                   = rhs.toarray()
       if self.domain_nb == 0 :
          b                   = b[1:]      
       elif self.domain_nb == 1:
          b                   = b[:-1]
       else:
          b                   = b[:]
       
       
       xkron               = self.lu.solve(b)       
       # ...
       x                   = np.zeros(V.nbasis)
       if self.domain_nb == 0 :
          x[1:]  = xkron[:]
       elif self.domain_nb == 1:
          x[:-1] = xkron[:]
       else:
          x[:] = xkron[:]
       		
       	
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
nelements   = 16
 
alpha_1       = 0.25#0.4
alpha_2        = 0.25
alpha_3       = 0.5
alpha_4        = 0.5	
alpha_5        = 0.75
alpha_6        = 0.75 
#overlap     = alpha - beta
xuh_0    = []
xuh_01   = []
xuh_02   = []
xuh_03   = []
iter_max = 111
S_DDM       = 1./(1-alpha_1)

#----------------------
#..... Initialisation
#----------------------
grids_0 = linspace(0., alpha_1, nelements+1)
# create the spline space for each direction
V_0     = SplineSpace(degree=degree, nelements= nelements, grid =grids_0)

grids_1 = linspace(alpha_2, alpha_3, nelements+1)
# create the spline space for each direction
V_1     = SplineSpace(degree=degree, nelements= nelements, grid =grids_1)

grids_2 = linspace(alpha_4,alpha_5 , nelements+1)
# create the spline space for each direction
V_2     = SplineSpace(degree=degree, nelements= nelements, grid =grids_2)

grids_3 = linspace(alpha_6, 1. , nelements+1)
# create the spline space for each direction
V_3     = SplineSpace(degree=degree, nelements= nelements, grid =grids_3)
#..... Initialisation
#--------------------------

Vt_0    = TensorSpace(V_0, V_1, V_2)
Vt_1    = TensorSpace(V_1, V_0, V_2)
Vt_2    = TensorSpace(V_2, V_1, V_3) 
Vt_3    = TensorSpace(V_3, V_2, V_1 )

DDM_0   = DDM_poisson( V_0, Vt_0,  S_DDM, 0, ovlp_value_left = alpha_1 , ovlp_value_right = 0.)
DDM_1   = DDM_poisson( V_1, Vt_1,  S_DDM, 2, ovlp_value_left = alpha_2, ovlp_value_right = alpha_3 )
DDM_2   = DDM_poisson( V_2, Vt_2,  S_DDM, 2, ovlp_value_left = alpha_4 ,  ovlp_value_right = alpha_5)
DDM_3   = DDM_poisson( V_3, Vt_3,  S_DDM, 1, ovlp_value_left = alpha_6 ,  ovlp_value_right = 1.)

# ... communication Dirichlet interface
u_0     = StencilVector(V_0.vector_space)
u_1     = StencilVector(V_1.vector_space)
u_2     = StencilVector(V_2.vector_space)
u_3     = StencilVector(V_3.vector_space)

u_00     = StencilVector(V_0.vector_space)
u_11     = StencilVector(V_1.vector_space)
u_22     = StencilVector(V_2.vector_space)
u_33     = StencilVector(V_3.vector_space)
print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve( u_11 , u_22 )
xuh_0.append(xuh)

u_1, xuh_1, l2_norm1, H1_norm1   = DDM_1.solve( u_00, u_22)
xuh_01.append(xuh_1)

u_2, xuh_2, l2_norm2, H1_norm2    = DDM_2.solve( u_11, u_33)
xuh_02.append(xuh_2)

u_3, xuh_3, l2_norm3, H1_norm3    = DDM_3.solve( u_22, u_11)
xuh_03.append(xuh_3)

u_00 = u_0
u_11 = u_1
u_22 = u_2
u_33 = u_3
l2_err = sqrt(l2_norm**2 + l2_norm1**2 + l2_norm2**2 +l2_norm3**2)
H1_err = sqrt(H1_norm**2+ H1_norm1**2 + H1_norm2**2 +H1_norm3**2)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))


plt.show()
for i in range(iter_max):
	u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve( u_1 , u_2 )
	u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve( u_11 , u_22 )
	xuh_0.append(xuh)

	u_1, xuh_1, l2_norm1, H1_norm1   = DDM_1.solve( u_00, u_22)
	xuh_01.append(xuh_1)

	u_2, xuh_2, l2_norm2, H1_norm2    = DDM_2.solve( u_11, u_33)
	xuh_02.append(xuh_2)

	u_3, xuh_3, l2_norm3, H1_norm3    = DDM_3.solve( u_22, u_11)
	xuh_03.append(xuh_3)

	u_00 = u_0
	u_11 = u_1
	u_22 = u_2
	u_33 = u_3
	l2_err = sqrt(l2_norm**2 + l2_norm1**2 + l2_norm2**2 +l2_norm3**2)
	H1_err = sqrt(H1_norm**2+ H1_norm1**2 + H1_norm2**2 +H1_norm3**2)
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

#---Compute a solution


nbpts = 100
plt.figure()
plot_field_1d(V_0.knots, V_0.degree, xuh, nx=101, color='b')
plot_field_1d(V_2.knots, V_2.degree, xuh_2, nx=101, color='r')
plot_field_1d(V_3.knots, V_3.degree, xuh_3, nx=101, color='m')
plot_field_1d(V_1.knots, V_1.degree, xuh_1, nx=101, color='p')
plt.show()

# # ........................................................
# ....................For a plot
# #.........................................................
if True :
	plt.figure()
	for i in range(iter_max):
		plot_field_1d(V_0.knots, V_0.degree, xuh_0[i],  nx=101, color='b')
		plot_field_1d(V_2.knots, V_2.degree, xuh_02[i], nx=101, color='p')
		plot_field_1d(V_3.knots, V_3.degree, xuh_03[i], nx=101, color='m')
		plot_field_1d(V_1.knots, V_1.degree, xuh_01[i], nx=101, color='r')
		
		plt.savefig('DDMp_sol_evol.png')
	plt.show()



