from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import least_square_Bspline

# .. Matrices in 1D ..
from gallery_section_06 import assemble_massmatrix1D
from gallery_section_06 import assemble_stiffnessmatrix1D

from gallery_section_06 import assemble_vector_ex01   
from gallery_section_06 import assemble_vector_ex10
from gallery_section_06 import assemble_vector_ex02

assemble_mass1D        = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_stiffness1D   = compile_kernel( assemble_stiffnessmatrix1D, arity=2)

assemble_rhs01         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_rhs10         = compile_kernel(assemble_vector_ex10, arity=1)
assemble_basis         = compile_kernel(assemble_vector_ex02, arity=1)

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

   def __init__(self, V1, V2, V, Vh, S_DDM, domain_nb, ovlp_value, xD, yD, u0_D, u1_D):
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

       Id                  = zeros((V1.nbasis-2,V1.nbasis-2))   
       if domain_nb == 0 :
          for i in range(0,V1.degree):
               Id[-V1.degree:,-i-1]   += S_DDM*(basis[:]*basis[i])
       else :
          for i in range(0,V1.degree):
               Id[:V1.degree,i]       += S_DDM*(basis[:]*basis[i])
       K                   = csr_matrix( kron(K1, M2) + kron(M1, K2) +kron(Id,M2))
       # ...
       self.lu             = sla.splu(csc_matrix(K))
       
       # ++++
       self.spaces         = [V1, V2, V, Vh]
       self.ovlp_value     = ovlp_value
       self.S_DDM          = S_DDM
       self.domain_nb      = domain_nb
       self.xD             = xD 
       self.yD             = yD 
       self.u0_D           = u0_D
       self.u1_D           = u1_D
   def solve(self, u0_d, u1_d, u0 = None, u1 = None):

       V1, V2, V,  Vh      = self.spaces[:]
       # ...
       u0                  = StencilVector(V.vector_space)
       u1                  = StencilVector(V.vector_space)  
         
       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs01( Vh, fields = [u0_d, self.u0_D], knots = True, value = [self.ovlp_value, self.S_DDM, self.domain_nb], out = rhs )
       b                   = (rhs.toarray()).reshape(V.nbasis)
       b                   = b[1:-1, 1:-1] 
       b                   = b.reshape((V1.nbasis-2)*(V2.nbasis-2))

       xkron               = self.lu.solve(b)       
       # ...
       x0                   = np.zeros(V.nbasis)
       x0[1:-1,1:-1]        = xkron.reshape((V1.nbasis-2, V2.nbasis-2))[:,:]
       x0[:,:]             += self.xD[:,:] 
       #...
       u0.from_array(V, x0)

       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs10( Vh, fields = [u1_d, self.u1_D], knots = True, value = [self.ovlp_value, self.S_DDM, self.domain_nb], out = rhs )
       b                   = (rhs.toarray()).reshape(V.nbasis)
       b                   = b[1:-1, 1:-1] 
       b                   = b.reshape((V1.nbasis-2)*(V2.nbasis-2))

       xkron               = self.lu.solve(b)       
       # ...
       x1                  = np.zeros(V.nbasis)
       x1[1:-1,1:-1]       = xkron.reshape((V1.nbasis-2, V2.nbasis-2))[:,:]
       x1[:,:]            += self.yD[:,:] 
       #...
       u1.from_array(V, x1)
       # ...
       return u0, u1, x0, x1

degree      = 2
quad_degree = degree + 1
nelements   = 16
# ... please take into account that : beta < alpha 
grid_g      = linspace(0, 2., 2*nelements-1)
grid_y      = linspace(0, 1., nelements+1)
alpha       = grid_g[nelements-1]
beta        = grid_g[nelements-1]
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 10
S_DDM       = 1./(beta)

# ....
#--------------------------
#..... Initialisation
#--------------------------

# ... Derscritization I. stand for integral
Igrid_0 = linspace(0, alpha, nelements+1)
grids_0 = grid_g[0:nelements+1]

Igrid_1 = linspace(beta, 2., nelements+1)
grids_1 = grid_g[nelements-2:]


# create the spline space for each direction
V0      = SplineSpace(degree=degree, nelements= nelements, grid = grids_0, nderiv = 2, quad_degree = quad_degree, sharing_grid = Igrid_0)
# create the spline space for each direction
V1      = SplineSpace(degree=degree, nelements= nelements, grid = grids_1, nderiv = 2, quad_degree = quad_degree, sharing_grid = Igrid_1)

# ... B-spline space in y direction
V2      = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree, sharing_grid = grid_y)

# ...
Vh_0    = TensorSpace(V0, V2)
Vh_1    = TensorSpace(V1, V2)

Vt_0    = TensorSpace(V0, V1, V2)
Vt_1    = TensorSpace(V1, V0, V2)

# .... computatyion of Dirichlet boundary conditon
u01_0   = StencilVector(Vh_0.vector_space)
u10_0   = StencilVector(Vh_0.vector_space)

xD_0    = np.zeros(Vh_0.nbasis)
yD_0    = np.zeros(Vh_0.nbasis)


sol_dx = lambda x,y : (0.2+0.8*y)*cos(0.5*pi*x)
sol_dy = lambda x,y : (0.2+0.8*y)*sin(0.5*pi*x)

# .. test 1
fx0 = lambda y :  sol_dx(0.,y) 
fy0 = lambda x :  sol_dx(0.,0.)
fy1 = lambda x :  sol_dx(0.,1.)

gx0 = lambda y :  -1
gy0 = lambda x :  x-1.
gy1 = lambda x :  x-1.

xD_0[0, : ]              = least_square_Bspline(V2.degree, V2.knots, fx0)
xD_0[:-1,0]              = least_square_Bspline(V0.degree, V0.knots, fy0)[:-1]
xD_0[:-1, V2.nbasis - 1] = least_square_Bspline(V0.degree, V0.knots, fy1)[:-1]

yD_0[0, : ]              = least_square_Bspline(V2.degree, V2.knots, gx0)
yD_0[:-1,0]              = least_square_Bspline(V0.degree, V0.knots, gy0)[:-1]
yD_0[:-1, V2.nbasis - 1] = least_square_Bspline(V0.degree, V0.knots, gy1)[:-1]

u01_0.from_array(Vh_0, xD_0)
u10_0.from_array(Vh_0, yD_0)

#...
u01_1   = StencilVector(Vh_1.vector_space)
u10_1   = StencilVector(Vh_1.vector_space)

xD_1    = np.zeros(Vh_1.nbasis)
yD_1    = np.zeros(Vh_1.nbasis)

# . test 1
fx0 = lambda y : sol_dx(0.,y) 
fx1 = lambda y : sol_dx(1.,y) 
fy0 = lambda x : sol_dx((x-1.),0.)
fy1 = lambda x : sol_dx((x-1.),1.)

gx0 = lambda y : sol_dy(0.,y) 
gx1 = lambda y : sol_dy(1.,y)
gy0 = lambda x : sol_dy((x-1.),0.)
gy1 = lambda x : sol_dy((x-1.),1.)

xD_1[V1.nbasis-1, : ]  = least_square_Bspline(V2.degree, V2.knots, fx1)
xD_1[:,0]              = least_square_Bspline(V1.degree, V1.knots, fy0)
xD_1[:, V2.nbasis - 1] = least_square_Bspline(V1.degree, V1.knots, fy1)

yD_1[V1.nbasis-1, : ]  = least_square_Bspline(V2.degree, V2.knots, gx1)
yD_1[:,0]              = least_square_Bspline(V1.degree, V1.knots, gy0)
yD_1[:, V2.nbasis - 1] = least_square_Bspline(V1.degree, V1.knots, gy1)
u01_1.from_array(Vh_1, xD_1)
u10_1.from_array(Vh_1, yD_1)

# .... Initiation of solvers
DDM_0  = DDM_poisson( V0, V2, Vh_0, Vt_0,  S_DDM, 0, alpha, xD_0, yD_0, u01_0, u10_0)
DDM_1  = DDM_poisson( V1, V2, Vh_1, Vt_1,  S_DDM, 1,  beta, xD_1, yD_1, u01_1, u10_1)

# ... communication interface
u0_0   = StencilVector(Vh_0.vector_space)
u1_0   = StencilVector(Vh_0.vector_space)
u0_1   = StencilVector(Vh_1.vector_space)
u1_1   = StencilVector(Vh_1.vector_space)

print('#---IN-UNIFORM--MESH')
u00_0, u00_1, xuh0_0, xuh1_0  = DDM_0.solve( u0_1, u1_1, u0 = u0_0, u1 = u1_0)
xuh_0.append(xuh0_0)
u0_1, u1_1, xuh0_1, xuh1_1    = DDM_1.solve( u0_0, u1_0, u0 = u0_0, u1 = u1_0)
xuh_01.append(xuh0_1)
u0_0   = u00_0
u1_0   = u00_1

for i in range(iter_max):
	#...
	u00_0, u00_1, xuh0_0, xuh1_0 = DDM_0.solve(u0_1, u1_1)
	xuh_0.append(xuh0_0)
	u0_1, u1_1, xuh0_1, xuh1_1   = DDM_1.solve(u0_0, u1_0)
	xuh_01.append(xuh0_1)
	u0_0 = u00_0
	u1_0 = u00_1
	print('iteration',i,'---x', np.max(np.absolute(xuh0_1[1,:]-xuh0_0[-2,:])))
	print('iteration',i,'---y', np.max(np.absolute(xuh1_1[1,:]-xuh1_0[-2,:])))

# -------------------------------------------------------
# ... gethering results & CONSTRUCTION OF C^1 solution
# -------------------------------------------------------

grid_x = linspace(0., 2., 2*nelements-1)
# create the spline space for each direction
E1     = SplineSpace(degree=degree, nelements= 2*nelements-1, grid = grid_x, nderiv = 2, quad_degree = quad_degree)
Eh     = TensorSpace(E1, V2)
# ...
e01    = StencilVector(Eh.vector_space)
e10    = StencilVector(Eh.vector_space)
x01    = np.zeros(Eh.nbasis)
x10    = np.zeros(Eh.nbasis)

x01[:V1.nbasis-2,:]         = xuh0_0[:-2,:]
x01[V1.nbasis-degree:,:]    = xuh0_1[2:,:]

x10[:V1.nbasis-2,:]         = xuh1_0[:-2,:]
x10[V1.nbasis-degree:,:]    = xuh1_1[2:,:]
e01.from_array(Eh, x01)
e10.from_array(Eh, x10)


# # ........................................................
# .................... Plot results
# #.........................................................

nbpts = 100
#---Solution in uniform mesh
wx, a, b, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x01 , Eh.knots, Eh.degree)
wy, c, d  = pyccel_sol_field_2d((nbpts,nbpts),  x10, Eh.knots, Eh.degree)[:-2]

Z = a*d-b*c
print( ' min of Jacobian in the intire unit square =', np.min(Z) )
print( ' max of Jacobian in the intire unit square =', np.max(Z) )

#-----adaptive mesh plot
figtitle  = 'Volumetric parametrization'

fig, axes = plt.subplots( 1, 2, figsize=[12,12], num=figtitle )
for ax in axes:
   ax.set_aspect('equal')

#axes[0].set_title( 'Physical domain ' )
for i in range(nbpts):
    phidx = wx[:,i]
    phidy = wy[:,i]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(nbpts):
    phidx = wx[i,:]
    phidy = wy[i,:]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)

#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
i=0
phidx = wx[:,i]
phidy = wy[:,i]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
i=nbpts-1
phidx = wx[:,i]
phidy = wy[:,i]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
#''
i=0
phidx = wx[i,:]
phidy = wy[i,:]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
i=nbpts-1
phidx = wx[i,:]
phidy = wy[i,:]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
#axes[0].axis('off')
axes[0].margins(0,0)
# ...
#axes[0].set_title( 'Physical domain ' )

for i in range(nbpts):
    phidx = wx[:,i]
    phidy = wy[:,i]

    axes[1].plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(nbpts):
    phidx = wx[i,:]
    phidy = wy[i,:]

    axes[1].plot(phidx, phidy, '-b', linewidth = 0.25)

axes[1].plot(e01.toarray(), e10.toarray(), 'ro')
axes[1].axis('off')
#axes[0].margins(0,0)
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.show()

# # ........................................................
# ....................For a plot
# #.........................................................
#nbpts = 100
xs, ys       = np.meshgrid(Igrid_0, grid_y)

ux, a, b     = pyccel_sol_field_2d(None,  xuh0_0, Vh_0.knots, Vh_0.degree, meshes =(xs, ys))
uy, c, d     = pyccel_sol_field_2d(None,  xuh1_0, Vh_0.knots, Vh_0.degree, meshes =(xs, ys))
Z0 = a*d-b*c

xs, ys       = np.meshgrid(Igrid_1, grid_y)

vx, a1, b1   = pyccel_sol_field_2d(None,  xuh0_1, Vh_1.knots, Vh_1.degree, meshes =(xs, ys))
vy, c1, d1   = pyccel_sol_field_2d(None,  xuh1_1, Vh_1.knots, Vh_1.degree, meshes =(xs, ys))

Z1 = a1*d1-b1*c1
print( ' min-max of Jacobian in the intire unit square for subd0=', np.min(Z0), np.max(Z0) )
print( ' min-max of Jacobian in the intire unit square for subd1=', np.min(Z1), np.max(Z1))

#-----adaptive mesh plot
figtitle  = 'Volumetric parametrization'

fig, axes = plt.subplots( 1, 2, figsize=[12,12], num=figtitle )
for ax in axes:
   ax.set_aspect('equal')

#axes[0].set_title( 'Physical domain ' )
for i in range(ux.shape[1]):
    phidx = ux[:,i]
    phidy = uy[:,i]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(ux.shape[0]):
    phidx = ux[i,:]
    phidy = uy[i,:]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)

for i in range(vx.shape[1]):
    phidx = vx[:,i]
    phidy = vy[:,i]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(vx.shape[0]):
    phidx = vx[i,:]
    phidy = vy[i,:]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)

#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
i=0
phidx = ux[:,i]
phidy = uy[:,i]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
i=ux.shape[1]-1
phidx = ux[:,i]
phidy = uy[:,i]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
#''
i=0
phidx = ux[i,:]
phidy = uy[i,:]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
i=ux.shape[0]-1
phidx = ux[i,:]
phidy = uy[i,:]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
#axes[0].axis('off')
axes[0].plot(u0_0.toarray()[:-V1.nbasis], u1_0.toarray()[:-V1.nbasis], 'ro')
axes[0].plot(u0_1.toarray()[V1.nbasis:], u1_1.toarray()[V1.nbasis:], 'ko')
#axes[0].axis('off')
axes[0].margins(0,0)
# ...
axes[1].set_title('Jacobian function')
im      = axes[1].contourf( wx, wy, Z, cmap= 'jet')
divider = make_axes_locatable(axes[1]) 
cax     = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im, cax=cax)
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('VP_harmonic_c1_mappings.png')
plt.show()

