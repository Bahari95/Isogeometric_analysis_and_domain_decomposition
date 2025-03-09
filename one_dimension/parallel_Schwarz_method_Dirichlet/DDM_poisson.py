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
from gallery_section_04 import assemble_vector_ex02   

assemble_Pr          = compile_kernel(assemble_vector_ex02, arity=1)
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

from tabulate import tabulate

#==============================================================================
#.......Poisson ALGORITHM
#==============================================================================
class poisson_DDM(object):

   def __init__(self, V1):
       '''
       We delete the first and the last spline function
       as a technic for applying Dirichlet boundary condition
       '''
       #..Stiffness and Mass matrix in 1D in the first deriction
       K1                  = assemble_stiffness1D(V1)
       K1                  = K1.tosparse()
       K1                  = K1.toarray()[1:-1,1:-1]
       K1                  = csr_matrix(K1)
       # ...
       self.lu             = sla.splu(csc_matrix(K1))
       self.V1             = V1

   def solve(self, V2, u, domain_nb, ovlp_value):
   
       # ... compute the the updated dirchlet boundary
       u_d                 = StencilVector(self.V1.vector_space)
       rhs                 = assemble_Pr(V2, fields = [u], knots = True, value = [ovlp_value]) 

       # ... update the position of dichlet boundary
       x                   = np.zeros(self.V1.nbasis)
       if domain_nb ==0:
            x[-1]          = rhs.toarray()[0]       
       else :
            x[0]           = rhs.toarray()[0]
       u_d.from_array(self.V1, x)

       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( self.V1, fields = [u_d] )
       b                   = rhs.toarray()
       b                   = b[1:-1]
       # ...
       xkron               = self.lu.solve(b)       
       # ...
       x[ 1:-1 ]           = xkron
       u.from_array(self.V1, x)
       # ...
       Norm                = assemble_norm_l2(self.V1, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 6
nelements   = 128


# ... please take into account that : beta < alpha 
alpha       = 0.5  #grids__[nelements//2+1]
beta        = 0.5 # grids__[nelements//2-1]
overlap     = alpha - beta
xuh_0    = []
xuh_01   = []
iter_max = 100

#----------------------
#..... Initialisation
#----------------------
grids_0 = linspace(0, alpha, nelements+1)
# create the spline space for each direction
V1_0    = SplineSpace(degree=degree, nelements= nelements, grid =grids_0)

grids_1 = linspace(beta, 1., nelements+1)
# create the spline space for each direction
V1_1    = SplineSpace(degree=degree, nelements= nelements, grid =grids_1)

#... Initialization of Poissson DDM solver
DDM_0   = poisson_DDM(V1_0)
DDM_1   = poisson_DDM(V1_1)

# ... communication Dirichlet interface
uh_0    = StencilVector(V1_0.vector_space)
uh_1    = StencilVector(V1_1.vector_space)

print(uh_0)
print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve(V1_1, uh_1, 0, alpha)
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1   = DDM_1.solve(V1_0, uh_0, 1, beta)
xuh_01.append(xuh_1)
# ...
uh_0.from_array(V1_0, xuh)
uh_1.from_array(V1_1, xuh_1)
# ...
l2_err = l2_norm + l2_norm1
H1_err = H1_norm + H1_norm1
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

for i in range(iter_max):
	# ... Dirichlezt boudndary condition in x = 0.75 and 0.25
	#...
	u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve(V1_1, uh_1, 0, alpha)
	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1   = DDM_1.solve(V1_0, uh_0, 1, beta)
	xuh_01.append(xuh_1)
	# ...
	uh_0.from_array(V1_0, xuh)
	uh_1.from_array(V1_1, xuh_1)

	# ...
	l2_err = l2_norm + l2_norm1
	H1_err = H1_norm + H1_norm1
	print('iteration {} <-----> L^2-error ={} -----> H^1-error = {}'.format(i, l2_err, H1_err))

#---Compute a solution
nbpts = 100
from simplines import plot_field_1d
if True :
	plt.figure()
	for i in range(iter_max):
		plot_field_1d(V1_0.knots, V1_0.degree, xuh_0[i],  nx=101, color='b')
		plot_field_1d(V1_1.knots, V1_1.degree, xuh_01[i], nx=101, color='r')
	plt.show()

#---Compute a solution

nbpts = 100
plt.figure()
plot_field_1d(V1_0.knots, V1_0.degree, xuh, nx=101, color='b')
plot_field_1d(V1_1.knots, V1_1.degree, xuh_1, nx=101, color='r')
plt.show()
{}
