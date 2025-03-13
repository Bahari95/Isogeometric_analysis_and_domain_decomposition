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
from   numpy                        import zeros, linalg, asarray, linspace, array
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


degree      = 6
nelements   = 16

grid = linspace(0., 1., nelements+1)

import numpy as np


def split_grid(X, N, r):
    """
    Split a grid X into N overlapping subgrids with exactly r shared elements.

    Parameters:
    X (numpy array or list): The grid points.
    N (int): Number of subgrids.
    r (int): Number of shared elements between consecutive subgrids.

    Returns:
    list: A list of sub-intervals (each as a tuple of start and end points).
    """
    p = len(X) // N  # Compute the approximate subgrid size
    grid_size = len(X)
    L = []

    L.append((X[0], X[p + r]))  # First subgrid

    for i in range(1, N - 1):
        L.append((X[p * i - r], X[(i + 1) * p + r]))  # Middle subgrids

    L.append((X[(N - 1) * p - r], X[-1]))  # Last subgrid including the last element

    return L



N = 7	   # Number of subgrids
r = 1   # Shared elements

subgrids = array(split_grid(grid, N, r))

print(subgrids)
print(grid)
#print(subgrids)


# cearte your grids
gridi = []
for i in range(N ):
	gridi.append(linspace(subgrids[i, 0 ], subgrids[i, 1 ], nelements+1))

gridi = array(gridi)


#----------------------
#..... Initialisation
#----------------------
sp = []
# create the spline space for each subdomain
for i in range(N ):
	sp.append(SplineSpace(degree=degree, nelements= nelements, grid =gridi[i]))

sp = array(sp)

spt = []
spt.append(TensorSpace(sp[0], sp[1], sp[2]))
for i in range(1, N-1 ):
	spt.append(TensorSpace(sp[i], sp[i-1], sp[i+1]))
spt.append(TensorSpace(sp[N-1], sp[N-2], sp[N-3]))
spt = array(spt)


uh0 = []
for i in range(N):
	uh0.append(StencilVector(sp[i].vector_space))	

lists = [[] for _ in range(N)]
listsL2_norm = []
listsH1_norm = []
L_DDM = []
# TODO
# TODO find optimal parameter over single multipatch
# TODO  if have N chain decomposition the algorithm must be converge at N iteration theorem  by Ferderic Nataf
L_DDM.append(1./(1-gridi[0][-1]))
for i in range(1, N-1):
	L_DDM.append(1./(1.-gridi[i][-1]))
L_DDM.append(1./(1-gridi[N-1][0]))
iter_max = 30 # must be converge at N iteration theorem  by Ferderic Nataf

DDM = [] 
#... Initialization of Poissson DDM solver
DDM.append(DDM_poisson( sp[0], spt[0],  L_DDM[0], 0 , ovlp_value_left = gridi[0][-1] , ovlp_value_right = gridi[0][r]))
for i in range(1,N-1):
	DDM.append(DDM_poisson( sp[i], spt[i],  L_DDM[i], 2 , ovlp_value_left = gridi[i][0] , ovlp_value_right = gridi[i][-1]))
	
DDM.append(DDM_poisson( sp[N-1], spt[N-1],  L_DDM[N-1], 1 , ovlp_value_left = gridi[N-1][0] , ovlp_value_right = 1.))
DDM = array (DDM)
	
u_0,   xuh, l2_norm, H1_norm     = DDM[0].solve(uh0[1], uh0[2])
lists[0].append(xuh)
listsL2_norm.append(l2_norm)
listsH1_norm.append(H1_norm)
#  domain 1 ( alpha_2 , alpha_3 )
for i in range(1, N-1):
	u_1, xuh_1, l2_norm1, H1_norm1   = DDM[i].solve(uh0[i-1], uh0[i+1])
	lists[i].append(xuh_1)
	listsL2_norm.append(l2_norm1)
	listsH1_norm.append(H1_norm1)
#  domain 1 ( alpha_4 , 1. )
u_2, xuh_2, l2_norm2, H1_norm2   = DDM[N-1].solve(uh0[N-2], uh0[N-3])
lists[N-1].append(xuh_2)
listsL2_norm.append(l2_norm2)
listsH1_norm.append(H1_norm2)
l2_err = 0.0
H1_err = 0.0
for i in range(N):

	l2_err+= listsL2_norm[i]**2 
	H1_err+= listsH1_norm[i]**2
print('-----> L^2-error ={} -----> H^1-error = {}'.format(sqrt(l2_err), sqrt(H1_err)))

for  i in range(N):
	uh0[i].from_array(sp[i], lists[i][0]) 
for j in range(iter_max):
	listsL2_norm = []
	listsH1_norm = []
	u_0,   xuh, l2_norm, H1_norm     = DDM[0].solve(uh0[1], uh0[2])
	
	lists[0].append(xuh)
	listsL2_norm.append(l2_norm)
	listsH1_norm.append(H1_norm)
	#  domain 1 ( alpha_2 , alpha_3 )
	for i in range(1, N-1):
		u_1, xuh_1, l2_norm1, H1_norm1   = DDM[i].solve(uh0[i-1], uh0[i+1])
		lists[i].append(xuh_1)
		listsL2_norm.append(l2_norm1)
		listsH1_norm.append(H1_norm1)
	
	#  domain 1 ( alpha_4 , 1. )
	u_2, xuh_2, l2_norm2, H1_norm2   = DDM[N-1].solve(uh0[N-2], uh0[N-3])
	lists[N-1].append(xuh_2)
	listsL2_norm.append(l2_norm2)
	listsH1_norm.append(H1_norm2)
	l2_err = 0.0
	H1_err = 0.0
	for i in range(N):

		l2_err+= listsL2_norm[i]**2 
		H1_err+= listsH1_norm[i]**2
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(sqrt(l2_err), sqrt(H1_err)))

	for  i in range(N):
		
		uh0[i].from_array(sp[i], lists[i][j+1]) 

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
	
	

