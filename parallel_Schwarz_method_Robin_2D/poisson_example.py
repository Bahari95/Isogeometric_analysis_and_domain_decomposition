from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import getGeometryMap
from simplines import prolongation_matrix
from simplines import least_square_Bspline
#---In Poisson equation
from gallery_section_04 import assemble_vector_un_ex01   
from gallery_section_04 import assemble_matrix_un_ex01 
from gallery_section_04 import assemble_norm_ex01     

assemble_stiffness2D = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_un_ex01, arity=1)
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
#==============================================================================
class DDM_poisson(object):
   def __init__(self, V, V_TOT,  u11_mpH, u12_mpH,  v11_mpH, v12_mpH, u_d, S_DDM, domain_nb, ovlp_value):

  

       # ++++
       stiffness           = assemble_stiffness2D(V, fields = [u11_mpH, u12_mpH], value = [domain_nb, S_DDM, ovlp_value] )
       if domain_nb == 0 :
           stiffness       = apply_dirichlet(V, stiffness, dirichlet = [[True, False],[True, True]])
       else :
           stiffness       = apply_dirichlet(V, stiffness, dirichlet = [[False, True],[True, True]])
       
       # ...
       M                   = stiffness.tosparse()
       self.lu             = sla.splu(csc_matrix(M))
       self.domain_nb      = domain_nb
       self.S_DDM          = S_DDM
       self.ovlp_value     = ovlp_value
       self.sp             = [V, V_TOT]
       self.u11_mpH        = u11_mpH 
       self.u12_mpH        = u12_mpH
       self.v11_mpH        = v11_mpH 
       self.v12_mpH        = v12_mpH
       self.u_d			   = u_d
       self.x_d			   = u_d.toarray().reshape(V.nbasis)
   def solve(self, u_np):
          
       V, V_TOT            = self.sp[:]
       # ...
       u                   = StencilVector(V.vector_space)
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = StencilVector(V.vector_space)
       rhs                 = assemble_rhs( V_TOT, fields = [self.u11_mpH, self.u12_mpH, self.u_d, self.v11_mpH, self.v12_mpH, u_np], knots = True, value = [self.domain_nb, self.S_DDM, self.ovlp_value], out = rhs )

       if self.domain_nb == 0 :
          rhs              = apply_dirichlet(V, rhs, dirichlet = [[True, False],[True, True]])
       else :
          rhs              = apply_dirichlet(V, rhs, dirichlet = [[False, True],[True, True]])
       b                   = rhs.toarray()
       # ...
       x                   = self.lu.solve(b)
       # ...
       x                   = x.reshape(V.nbasis) + self.x_d
       #...
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[self.u11_mpH, self.u12_mpH, u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1
nelements   = 128 # please take it as factor of 16 

#.. Parameterisation of the domain and refinement

#--------------------------------------------------------------
#..... Initialisation and computing optimal mapping for 16*16
#--------------------------------------------------------------
#...
# mapping F1 and F2 from the reference element to the physical element
# F1 [0,1]x[0,1] -> [0,1]x[0,0.5] / interface 0.5 C^0, C^1 natural ( C^{p-1} ? p>3 )
# C^1 deux facons de faire : C^0 dans les bases, C^1 dans les points de controls 
# #// C^1 dans les bases bF1(n-1) et bF1(n) = bF2(0) et bF2(1) sont les memes
# F2 [0,1]x[0,1] -> [0,1]x[0.5,1]
# create the spline space for each direction
VH1       = SplineSpace(degree=degree, nelements=16)
VH1       = TensorSpace(VH1, VH1)# before refinement

Vh1       = SplineSpace(degree=degree, nelements=nelements)
Vh1       = TensorSpace(Vh1, Vh1)# after refinement
#----------------------------------------
#..... Parameterization from 16*16 elements
#----------------------------------------
# ... Circle :  patch 1


#geometry  = '../parallel_Schwarz_method_Robin_2D/annuls1.xml'
#geometry = '../parallel_Schwarz_method_Robin_2D/circle_ove1.xml'


geometry  = '../parallel_Schwarz_method_Robin_2D/Annalus1.xml'
#geometry  = '../parallel_Schwarz_method_Robin_2D/Annulus_over1.xml'

print('#---IN-UNIFORM--MESH-Poisson equation patch 1', geometry)


#Annuls : patch 1


# ... Assembling mapping
mp1       = getGeometryMap(geometry,0)
xmp1      = zeros(VH1.nbasis)
ymp1      = zeros(VH1.nbasis)

xmp1[:,:], ymp1[:,:] =  mp1.coefs()

#geometry = '../parallel_Schwarz_method_Robin_2D/circle2.xml'
#geometry = '../parallel_Schwarz_method_Robin_2D/circle_ove2.xml'

geometry = '../parallel_Schwarz_method_Robin_2D/Annalus2.xml'
#geometry  = '../parallel_Schwarz_method_Robin_2D/Annulus_over2.xml'
print('#---IN-UNIFORM--MESH-Poisson equation patch 2', geometry)


# ... Assembling mapping
mp2             = getGeometryMap(geometry,0) # second part
xmp2 = zeros(VH1.nbasis)
ymp2 = zeros(VH1.nbasis)

xmp2[:,:], ymp2[:,:] =  mp2.coefs()
#xmp2[0,:] = xmp1[-1,:] 
#ymp2[0,:] = ymp1[-1,:]
# ... C1 continuity garad(F1 = F2)  = garad(F2)  interface
#xmp2[1,:] = 2.*xmp1[-1,:] - xmp1[-2,:]
#ymp2[1,:] = 2.*ymp1[-1,:] - ymp1[-2,:] 
#... Prolongation by knot insertion
M_mp      = prolongation_matrix(VH1, Vh1)

xmp1      = (M_mp.dot(xmp1.reshape(VH1.nbasis[0]*VH1.nbasis[1]))).reshape(Vh1.nbasis)
ymp1      = (M_mp.dot(ymp1.reshape(VH1.nbasis[0]*VH1.nbasis[1]))).reshape(Vh1.nbasis)
xmp2      = (M_mp.dot(xmp2.reshape(VH1.nbasis[0]*VH1.nbasis[1]))).reshape(Vh1.nbasis)
ymp2      = (M_mp.dot(ymp2.reshape(VH1.nbasis[0]*VH1.nbasis[1]))).reshape(Vh1.nbasis)

#-------------++++++++++++++++-------------------------------------------------

fig =plt.figure() 
for i in range(Vh1.nbasis[1]):
   phidx = xmp1[:,i]
   phidy = ymp1[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
for i in range(Vh1.nbasis[0]):
   phidx = xmp1[i,:]
   phidy = ymp1[i,:]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
phidx = xmp1[:,0]
phidy = ymp1[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = 'patch 2 $Im([0,1]^2_{y=0})$')
# ...
phidx = xmp1[:,Vh1.nbasis[1]-1]
phidy = ymp1[:,Vh1.nbasis[1]-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = 'patch 2 $Im([0,1]^2_{y=1})$')
#''

phidx = xmp1[0,:]
phidy = ymp1[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = 'patch 2 $Im([0,1]^2_{x=0})$')
# ...
phidx = xmp1[Vh1.nbasis[1]-1,:]
phidy = ymp1[Vh1.nbasis[1]-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = 'patch 2 $Im([0,1]^2_{x=1}$)')



'''
num_points = 100

# Generate (x, y) values in the unit square (0,1) x (0,1)
x_vals = np.linspace(0, 1, num_points)
y_vals = np.linspace(0, 1, num_points)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
f1 = lambda x,y : (x*0.5+1)*cos(pi/2 * (y))
f2 = lambda x,y : (x*.5+1)*sin(pi/2 * (y))
# Apply transformation
X_transformed = f1(X_grid,Y_grid )
Y_transformed = f2(X_grid,Y_grid )


# Plot the transformed points
plt.figure(figsize=(6, 6))


for i in range(num_points):
   phidx = X_transformed[:,i]
   phidy = Y_transformed[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
for i in range(num_points):
   phidx = X_transformed[i,:]
   phidy = Y_transformed[i,:]
   plt.plot(phidx, phidy, '-b', linewidth = .3)

plt.show()

''' 
   
for i in range(Vh1.nbasis[1]):
   phidx = xmp2[:,i]
   phidy = ymp2[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
for i in range(Vh1.nbasis[0]):
   phidx = xmp2[i,:]
   phidy = ymp2[i,:]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
   
#.. Plot the surface in the first patch 1

#.. Plot the surface in the second patch 2

phidx = xmp2[:,0]
phidy = ymp2[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = 'patch 1 $Im([0,1]^2_{y=0})$')
# ...
phidx = xmp2[:,Vh1.nbasis[1]-1]
phidy = ymp2[:,Vh1.nbasis[1]-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = 'patch 1 $Im([0,1]^2_{y=1})$')
#''
phidx = xmp2[0,:]
phidy = ymp2[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = 'patch 1 $Im([0,1]^2_{x=0})$')
# ...
phidx = xmp2[Vh1.nbasis[1]-1,:]
phidy = ymp2[Vh1.nbasis[1]-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = 'patch 1 $Im([0,1]^2_{x=1}$)')

plt.legend()
#plt.scatter(xmp1[Vh1.nbasis[1]-1,:],ymp1[Vh1.nbasis[1]-1,:], color= 'black', linewidths=3.)
#plt.scatter(xmp1[Vh1.nbasis[1]-1,Vh1.nbasis[1]-1],ymp1[Vh1.nbasis[1]-1,Vh1.nbasis[1]-1],  color= 'black', linewidths=3.)
plt.show() 


#--------------------------------------------------------------

#...End of parameterisation
#--------------------------------------------------------------
# ... please take into account that : beta < alpha 
alpha       = 0.5 # fixed by the geometry parameterization
beta        = 0.5# fixed by the geometry parameterization
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 100
S_DDM       = 1./alpha**2 #alpha/(nelements+1)
#.. test 0
u_exact   = lambda x, y : sin(2.*pi*x)*sin(2.*pi*y)
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

# ... mappings as stencil vector
u11_mpH        = StencilVector(V_0.vector_space)
u12_mpH        = StencilVector(V_0.vector_space)
u11_mpH.from_array(V_0, xmp1)
u12_mpH.from_array(V_0, ymp1)
# ...
v11_mpH        = StencilVector(V_1.vector_space)
v12_mpH        = StencilVector(V_1.vector_space)
v11_mpH.from_array(V_1, xmp2)
v12_mpH.from_array(V_1, ymp2)

# --- Initialization domain in left
domain_nb = 0

n_dir      = V1_0.nbasis + V2_0.nbasis+100
sX         = pyccel_sol_field_2d((n_dir,n_dir),  xmp1 , V_0.knots, V_0.degree)[0]
sY         = pyccel_sol_field_2d((n_dir,n_dir),  ymp1 , V_0.knots, V_0.degree)[0]
u_d        = StencilVector(V_0.vector_space)
x_d        = np.zeros(V_0.nbasis)
x_d[0, : ] = least_square_Bspline(V2_0.degree, V2_0.knots, u_exact(sX[0, :], sY[ 0,:]), m= n_dir)
#x_d[-1, :] = least_square_Bspline(V2_0.degree, V2_0.knots, u_exact(sX[-1,:], sY[-1,:]), m= n_dir)
x_d[:,0]   = least_square_Bspline(V1_0.degree, V1_0.knots, u_exact(sX[:, 0], sY[:, 0]), m= n_dir)
x_d[:, -1] = least_square_Bspline(V1_0.degree, V1_0.knots, u_exact(sX[:,-1], sY[:,-1]), m= n_dir)
u_d.from_array(V_0, x_d)

P0      = DDM_poisson(V_0, Vt_0, u11_mpH, u12_mpH, v11_mpH, v12_mpH, u_d, S_DDM, domain_nb, alpha )

# --- Initialization domain in right
domain_nb = 1
sX         = pyccel_sol_field_2d((n_dir,n_dir),  xmp2 , V_1.knots, V_1.degree)[0]
sY         = pyccel_sol_field_2d((n_dir,n_dir),  ymp2 , V_1.knots, V_1.degree)[0]
u_d        = StencilVector(V_1.vector_space)
x_d        = np.zeros(V_1.nbasis)
#x_d[0, : ] = least_square_Bspline(V2_1.degree, V2_1.knots, u_exact(sX[0, :], sY[ 0,:]), m= n_dir)
x_d[-1, :] = least_square_Bspline(V2_1.degree, V2_1.knots, u_exact(sX[-1,:], sY[-1,:]), m= n_dir)
x_d[:,0]   = least_square_Bspline(V1_1.degree, V1_1.knots, u_exact(sX[:, 0], sY[:, 0]), m= n_dir)
x_d[:, -1] = least_square_Bspline(V1_1.degree, V1_1.knots, u_exact(sX[:,-1], sY[:,-1]), m= n_dir)
u_d.from_array(V_1, x_d)
P1      = DDM_poisson(V_1, Vt_1, v11_mpH, v12_mpH, u11_mpH, u12_mpH, u_d, S_DDM, domain_nb, beta  )

# ... communication Solution interface
u_00    = StencilVector(V_0.vector_space)
u_1     = StencilVector(V_1.vector_space)

print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm, H1_norm     = P0.solve(u_1)
u_00 = u_0
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1   = P1.solve(u_0)
xuh_01.append(xuh_1)
l2_err = sqrt(l2_norm**2 + l2_norm1**2)
H1_err = sqrt(H1_norm**2 + H1_norm1**2)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

for i in range(iter_max):
	#...
	u_0, xuh, l2_norm, H1_norm     = P0.solve(u_1)
	print('.',i)
	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1 = P1.solve(u_00)
	print('.',i)
	xuh_01.append(xuh_1)
	u_00   = u_0
	if abs(l2_err - l2_norm - l2_norm1) <=1e-10:
		iter_max = i+1
		print(iter_max)
		break
	l2_err = l2_norm + l2_norm1
	H1_err = H1_norm + H1_norm1
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

#---Compute a solution
nbpts = 100
# # ........................................................
# ....................For a plot
# #.........................................................
if True :
	# --- First Subdomain ---
	#---Compute a solution
	u, ux, uy, X, Y               = pyccel_sol_field_2d((nbpts, nbpts),  xuh,   V_0.knots, V_0.degree)
	# ...
	u_1, ux_1, uy_1, X_1, Y_1     = pyccel_sol_field_2d((nbpts, nbpts),  xuh_1,   V_1.knots, V_1.degree)
	u_0  = []
	u_01 = []
	for i in range(iter_max):
	    u_0.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_0[i],   V_0.knots, V_0.degree)[0][:,50])
	    u_01.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_01[i],   V_1.knots, V_1.degree)[0][:,50])
	plt.figure() 
	plt.axes().set_aspect('equal')
	plt.subplot(121)
	for i in range(iter_max-1):
	     plt.plot(X[:,50], u_0[i], '-k', linewidth = 1.)
	     plt.plot(X_1[:,50], u_01[i], '-k', linewidth = 1.) 
	plt.plot(X[:,50], u_0[i+1], '-k', linewidth = 1., label='$\mathbf{Un_0-iter(i)}$')
	plt.plot(X_1[:,50], u_01[i+1], '-k', linewidth = 1., label='$\mathbf{Un_1-iter(i)}$')
	plt.legend()
	plt.grid(True)  
	plt.subplot(122)
	plt.plot(X[:,50], u[:,50],  '--or', label = '$\mathbf{Un_0-iter-max}$' )
	plt.plot(X_1[:,50], u_1[:,50],  '--om', label = '$\mathbf{Un_1-iter-max}$')

	plt.legend()
	plt.grid(True)  
	plt.savefig('behvoir_between_two_patches.png')

	      
	u1, ux, uy, X, Y = pyccel_sol_field_2d((nbpts, nbpts), xuh, V_0.knots, V_0.degree)
	F1_1 = pyccel_sol_field_2d((nbpts, nbpts), xmp1, V_0.knots, V_0.degree)[0]
	F2_1 = pyccel_sol_field_2d((nbpts, nbpts), ymp1, V_0.knots, V_0.degree)[0]

	# --- Second Subdomain ---
	u2, ux, uy, X, Y = pyccel_sol_field_2d((nbpts, nbpts), xuh_1, V_1.knots, V_1.degree)
	F1_2 = pyccel_sol_field_2d((nbpts, nbpts), xmp2, V_1.knots, V_1.degree)[0]
	F2_2 = pyccel_sol_field_2d((nbpts, nbpts), ymp2, V_1.knots, V_1.degree)[0]

	# --- Compute Global Color Levels ---
	u_min = min(np.min(u1), np.min(u2))
	u_max = max(np.max(u1), np.max(u2))
	levels = np.linspace(u_min, u_max, 100)  # Uniform levels for both plots

	# --- Create Figure ---
	fig, axes = plt.subplots(figsize=(8, 6))

	# --- Contour Plot for First Subdomain ---
	im1 = axes.contourf(F1_1, F2_1, u1, levels, cmap='jet')

	# --- Contour Plot for Second Subdomain ---
	im2 = axes.contourf(F1_2, F2_2, u2, levels, cmap='jet')

	# --- Colorbar ---
	divider = make_axes_locatable(axes)
	cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
	cbar = plt.colorbar(im2, cax=cax)
	cbar.ax.tick_params(labelsize=15)
	cbar.ax.yaxis.label.set_fontweight('bold')

	# --- Formatting ---
	axes.set_title("Solution the in whole domain ", fontweight='bold')
	for label in axes.get_xticklabels() + axes.get_yticklabels():
	    label.set_fontweight('bold')

	fig.tight_layout()
	plt.savefig('2patch.png')
	plt.show()
