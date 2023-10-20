from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import plot_field_1d
from simplines import pyccel_sol_field_2d

# .. Matrices in 1D ..
from gallery_section_05 import assemble_massmatrix1D
from gallery_section_05 import assemble_stiffnessmatrix1D
from gallery_section_05 import assemble_vector_ex01   
from gallery_section_05 import assemble_vector_ex02
from gallery_section_05 import assemble_norm_ex01     

assemble_mass1D        = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_stiffness1D   = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_rhs           = compile_kernel(assemble_vector_ex01, arity=1)
assemble_basis         = compile_kernel(assemble_vector_ex02, arity=1)
assemble_norm_l2       = compile_kernel(assemble_norm_ex01, arity=1)

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

   def __init__(self, V1, V2, V, Vh, S_DDM, domain_nb, ovlp_value):
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition
       
       basis              = assemble_basis(V1, knots = True, value = [ovlp_value])
       basis              = basis.toarray()[0:V1.degree]

       #..Stiffness and Mass matrix in 1D in the first deriction
       K1                 = assemble_stiffness1D(V1)
       K1                 = K1.tosparse()
       K1                 = K1.toarray()[1:-1,1:-1]

       #___
       M1                 = assemble_mass1D(V1)
       M1                 = M1.tosparse()
       M1                 = M1.toarray()[1:-1,1:-1]

       #..Stiffness and Mass matrix in 1D in the first deriction
       K2                 = assemble_stiffness1D(V2)
       K2                 = K2.tosparse()
       K2                 = K2.toarray()[1:-1,1:-1]

       #___
       M2                 = assemble_mass1D(V2)
       M2                 = M2.tosparse()
       M2                 = M2.toarray()[1:-1,1:-1]
              
       if domain_nb == 0 :
          for i in range(0,V1.degree):
               K1[-V1.degree:,-i-1]   += S_DDM*(basis[:]*basis[i])
       else :
          for i in range(0,V1.degree):
               K1[:V1.degree,i]       += S_DDM*(basis[:]*basis[i])
       K                   = csr_matrix( kron(K1, M2) + kron(M1, K2) )

       # ...
       self.lu             = sla.splu(csc_matrix(K))
       
       # ++++
       self.spaces         = [V1, V2, V, Vh]
       self.ovlp_value     = ovlp_value
       self.S_DDM          = S_DDM
       self.domain_nb      = domain_nb
   def solve(self, u_d):

       V1, V2, V,  Vh      = self.spaces[:]

       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs( Vh, fields = [u_d], knots = True, value = [self.ovlp_value, self.S_DDM, self.domain_nb], out = rhs )
       b                   = (rhs.toarray()).reshape(V.nbasis)
       b                   = b[1:-1, 1:-1] 
       b                   = b.reshape((V1.nbasis-2)*(V2.nbasis-2))

       xkron               = self.lu.solve(b)       
       # ...
       x                   = np.zeros(V.nbasis)
       x[1:-1,1:-1]        = xkron.reshape((V1.nbasis-2, V2.nbasis-2))[:,:]
       #...
       u                   = StencilVector(V.vector_space)
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 3
quad_degree = degree + 1
nelements   = 128
# .. interval boundary values
left_v      = 0.
right_v     = 1.

# ... please take into account that : beta < alpha 
grid_g      = linspace(left_v, right_v, 2*nelements-1)
grid_y      = linspace(0, 1., nelements+1)
alpha       = grid_g[nelements-1]
beta        = grid_g[nelements-1]
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 10
S_DDM       = 1./(beta)

#--------------------------
#..... Initialisation
#--------------------------

#... Derscritization I. stand for integral

Igrid_0 = linspace(left_v, alpha, nelements+1)
grids_0 = grid_g[0:nelements+1]

Igrid_1 = linspace(beta, right_v, nelements+1)
grids_1 = grid_g[nelements-2:]

# create the spline space for each direction
V_0     = SplineSpace(degree=degree, nelements= nelements, grid = grids_0, nderiv = 2, quad_degree = quad_degree, sharing_grid = Igrid_0)
# create the spline space for each direction
V_1     = SplineSpace(degree=degree, nelements= nelements, grid = grids_1, nderiv = 2, quad_degree = quad_degree, sharing_grid = Igrid_1)

# ... B-spline space in y direction
V_2     = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree, sharing_grid = grid_y)

# ...
Vh_0    = TensorSpace(V_0, V_2)
Vh_1    = TensorSpace(V_1, V_2)

Vt_0    = TensorSpace(V_0, V_1, V_2)
Vt_1    = TensorSpace(V_1, V_0, V_2)

DDM_0   = DDM_poisson( V_0, V_2, Vh_0, Vt_0,  S_DDM, 0, alpha )
DDM_1   = DDM_poisson( V_1, V_2, Vh_1, Vt_1,  S_DDM, 1, beta )

# ... communication Dirichlet interface
u_00    = StencilVector(Vh_0.vector_space)
u_1     = StencilVector(Vh_1.vector_space)

print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = DDM_0.solve( u_1)
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1   = DDM_1.solve( u_00)
xuh_01.append(xuh_1)
u_00   = u_0
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
nbpts = 100
plt.figure()
plot_field_1d(V_0.knots, V_0.degree, xuh[:,20],   nx=nbpts, color='b', xmin = left_v, xmax = alpha)
plot_field_1d(V_1.knots, V_1.degree, xuh_1[:,20], nx=nbpts, color='r', xmin = beta, xmax = right_v)
plt.show()

# # ........................................................
# ....................For a plot
# #.........................................................
if True :
	plt.figure()
	for i in range(iter_max):
		r = np.round(np.random.rand(),1)
		g = np.round(np.random.rand(),1)
		b = np.round(np.random.rand(),1)
		# ...
		plot_field_1d(V_0.knots, V_0.degree, xuh_0[i][:,20],  nx= nbpts, color=[r,b,g], xmin = left_v, xmax =alpha)
		plot_field_1d(V_1.knots, V_1.degree, xuh_01[i][:,20], nx= nbpts, color=[r,b,g], xmin = beta, xmax =right_v)
	plt.show()

# -------------------------------------------------------
# ... gethering results & CONSTRUCTION OF C^{p-1} solution
# -------------------------------------------------------

grid_x = linspace(left_v, right_v, 2*nelements-1)
# create the spline space for each direction
E1     = SplineSpace(degree=degree, nelements= 2*nelements-1, grid = grid_x, nderiv = 2, quad_degree = quad_degree)
Eh     = TensorSpace(E1, V_2)
# ...
eh     = StencilVector(Eh.vector_space)
xh     = np.zeros(Eh.nbasis)

xh[:V_1.nbasis-2,:]         = xuh[:-2,:]
xh[V_1.nbasis-degree:,:]    = xuh_1[2:,:]
eh.from_array(Eh, xh)

# # ........................................................
# .................... 
# #.........................................................
#---Solution in uniform mesh
w, wx, wy, X,Y = pyccel_sol_field_2d((nbpts,nbpts),  xh, Eh.knots, Eh.degree)

#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
figtitle  = 'mesh adaptation'

fig, axes = plt.subplots( 1, 2, figsize=[12,12], gridspec_kw={'width_ratios': [2.75, 2]}, num=figtitle )
for ax in axes:
   ax.set_aspect('equal')

axes[0].set_title('Ap. Sol. in the entire domain')
ima     = axes[0].contourf( X, Y, w, cmap= 'plasma')
divider = make_axes_locatable(axes[0]) 
caxe    = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(ima, cax=caxe)

axes[1].set_title('Ap. first deriv/dy')
im      = axes[1].contourf( X, Y, wy, cmap= 'jet')
divider = make_axes_locatable(axes[1]) 
cax     = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im, cax=cax)
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('example.png')
plt.show()

