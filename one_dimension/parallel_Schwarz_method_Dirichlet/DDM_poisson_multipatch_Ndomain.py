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
from   numpy                        import zeros, linalg, asarray, linspace, array
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

   def solve(self, V2, u1, domain_nb, ovlp_value_left, V3=None, u2=None, ovlp_value_right=None):
   
       # ... compute the the updated dirchlet boundary
       u_d                   = StencilVector(self.V1.vector_space)
       x                     = np.zeros(self.V1.nbasis)
       if ovlp_value_right is None:
          rhs_l                 = assemble_Pr(V2, fields = [u1], knots = True, value = [ovlp_value_left]) 
       	  if domain_nb ==0:
       	    
             x[-1]          = rhs_l.toarray()[0]       
       	  else :

             x[0]           = rhs_l.toarray()[0]
            
       else:
          rhs_l                 = assemble_Pr(V2, fields = [u1], knots = True, value = [ovlp_value_left]) 
          rhs_r                 = assemble_Pr(V3, fields = [u2], knots = True, value = [ovlp_value_right])
          x[0]           = rhs_l.toarray()[0]
          x[-1]          = rhs_r.toarray()[0]  
       	  

       # ... update the position of dichlet boundary
       
       u_d.from_array(self.V1, x)

       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( self.V1, fields = [u_d] )
       b                   = rhs.toarray()
       b                   = b[1:-1]
       # ...
       xkron               = self.lu.solve(b)       
       # ...
       x[ 1:-1 ]           = xkron
       u_d.from_array(self.V1, x)
       # ...
       Norm                = assemble_norm_l2(self.V1, fields=[u_d]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u_d, x, l2_norm, H1_norm

degree      = 2
nelements   = 16

iter_max = 500

grid = linspace(0., 1., nelements+1)

import numpy as np

def split_interval_from_grid(grid, N, r):
    """
    Split a predefined grid into N overlapping sub-intervals with overlap of r elements.
    
    Parameters:
    grid (list or numpy array): Predefined grid points
    N (int): Number of sub-intervals
    r (int): Number of overlapping elements
    
    Returns:
    list: List of tuples representing the overlapping sub-intervals
    """
    
    if len(grid) < 2 or N < 1 or r < 0:
        raise ValueError("Grid must have at least 2 points, N must be at least 1, and r must be non-negative.")
    
    indices = np.linspace(0, len(grid) - 1, N + 1, dtype=int)  # Ensure uniform distribution
    intervals = []
    
    for i in range(N):
        start_idx = indices[i]
        end_idx = min(indices[i + 1] + r, len(grid) - 1)  # Ensure overlap but don't exceed last index
        intervals.append((grid[start_idx], grid[end_idx]))
    
    return intervals


# Example usage:
	
N = 12  # Number of sub-intervals
r = 1  # Overlap elements
intervals = asarray(split_interval_from_grid(grid, N, r))

# cearte your grids
gridi = []
for i in range(N ):
	gridi.append(linspace(intervals[i, 0 ], intervals[i, 1 ], nelements+1))

gridi = array(gridi)


#----------------------
#..... Initialisation
#----------------------
sp = []
# create the spline space for each direction
for i in range(N ):
	sp.append(SplineSpace(degree=degree, nelements= nelements, grid =gridi[i]))

sp = array(sp)

DDM = [] 
#... Initialization of Poissson DDM solver
for i in range(N):
	DDM.append(poisson_DDM(sp[i]))
DDM = array (DDM)

uh = []
for i in range(N):
	uh.append(StencilVector(sp[i].vector_space))


lists = [[] for _ in range(N)]
listsL2_norm = []
listsH1_norm = []



print('#---IN-UNIFORM--MESH')
# domain0 ( 0. , alpha_1 )
u_0,   xuh, l2_norm, H1_norm     = DDM[0].solve(V2 = sp[1], u1 = uh[1], domain_nb = 0, ovlp_value_left= gridi[0][-1])
lists[0].append(xuh)
listsL2_norm.append(l2_norm)
listsH1_norm.append(H1_norm)
#  domain 1 ( alpha_2 , alpha_3 )
for i in range(1, N-1):
	u_1, xuh_1, l2_norm1, H1_norm1   = DDM[i].solve(V2 = sp[i-1], u1 =  uh[i-1], domain_nb =  i, ovlp_value_left = gridi[i][0], V3 =  sp[i+1], u2 = uh[i+1], ovlp_value_right =  gridi[i][-1])
	lists[i].append(xuh_1)
	listsL2_norm.append(l2_norm1)
	listsH1_norm.append(H1_norm1)
#  domain 1 ( alpha_4 , 1. )
u_2, xuh_2, l2_norm2, H1_norm2   = DDM[N-1].solve(V2 = sp[N-2], u1 = uh[N-2], domain_nb =  1, ovlp_value_left = gridi[N-1][0])
lists[N-1].append(xuh_2)
listsL2_norm.append(l2_norm2)
listsH1_norm.append(H1_norm2)
# ...
for i in range(N):
	uh[i].from_array(sp[i], lists[i][0])

	
# ...
l2_err = 0.0
H1_err = 0.0
for i in range(N):

	l2_err+= listsL2_norm[i]**2 
	H1_err+= listsH1_norm[i]**2

print('iteration {}-----> L^2-error ={} -----> H^1-error = {}'.format(0, sqrt(l2_err), sqrt(H1_err)))

for j in range(iter_max):
	listsL2_norm = []
	listsH1_norm = []



	# domain0 ( 0. , alpha_1 )
	u_0,   xuh, l2_norm, H1_norm     = DDM[0].solve(V2 = sp[1], u1 = uh[1], domain_nb = 0, ovlp_value_left= gridi[0][-1])
	lists[0].append(xuh)
	listsL2_norm.append(l2_norm)
	listsH1_norm.append(H1_norm)
	#  domain 1 ( alpha_2 , alpha_3 )
	for i in range(1, N-1):
		u_1, xuh_1, l2_norm1, H1_norm1   = DDM[i].solve(V2 = sp[i-1], u1 =  uh[i-1], domain_nb =  i, ovlp_value_left = gridi[i][0], V3 =  sp[i+1], u2 = uh[i+1], ovlp_value_right =  gridi[i][-1])
		lists[i].append(xuh_1)
		listsL2_norm.append(l2_norm1)
		listsH1_norm.append(H1_norm1)
	#  domain 1 ( alpha_4 , 1. )
	u_2, xuh_2, l2_norm2, H1_norm2   = DDM[N-1].solve(V2 = sp[N-2], u1 = uh[N-2], domain_nb =  1, ovlp_value_left = gridi[N-1][0])
	lists[N-1].append(xuh_2)
	listsL2_norm.append(l2_norm2)
	listsH1_norm.append(H1_norm2)
	# ...
	for i in range(N):
		uh[i].from_array(sp[i], lists[i][j+1])
		
	# ...
	l2_err = 0.0
	H1_err = 0.0
	for i in range(N):

		l2_err+= listsL2_norm[i]**2 
		H1_err+= listsH1_norm[i]**2

	print('iteration {}-----> L^2-error ={} -----> H^1-error = {}'.format(j, sqrt(l2_err), sqrt(H1_err)))


#---Compute a solution

#---Compute a solution
from simplines import plot_field_1d
nbpts = 100
plt.figure()
for i in range(N):
	plot_field_1d(sp[i].knots, sp[i].degree, lists[i][-1], nx=101, color='b')
	
plt.show()

# # ........................................................
# ....................For a plot
# #.........................................................

if True :
	plt.figure()
	for i in range(iter_max):
		
		for j in range(N):
			plot_field_1d(sp[j].knots, sp[j].degree, lists[j][i],  nx=101, color='b')
		
		
		plt.savefig('DDMp_sol_evol12.png')
	plt.show()


