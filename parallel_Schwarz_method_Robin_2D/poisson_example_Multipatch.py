from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import getGeometryMap
from simplines import prolongation_matrix
from simplines import least_square_Bspline
from simplines import plot_MeshMultipatch
#---In Poisson equation
from gallery_section_04_Multipatch import assemble_vector_un_ex01   
from gallery_section_04_Multipatch import assemble_matrix_un_ex01 
from gallery_section_04_Multipatch import assemble_norm_ex01     

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
from   numpy                        import zeros, linalg, asarray, linspace,dot
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2, cosh
from   tabulate                     import tabulate
import numpy                        as     np
import timeit
import time

#==============================================================================
#.......Poisson ALGORITHM
#==============================================================================
class DDM_poisson(object):
	def __init__(self, V, V_TOT,  u11_mpH, u12_mpH,  v11_mpH, v12_mpH, w11_mpH, w12_mpH , u_d, S_DDM, domain_nb, ovlp_value_left, ovlp_value_right):
		# ++++
		stiffness           = assemble_stiffness2D(V, fields = [u11_mpH, u12_mpH], value = [domain_nb, S_DDM, ovlp_value_left] )
		if domain_nb == 0 :
			stiffness       = apply_dirichlet(V, stiffness, dirichlet = [[True, False],[True, True]])
		elif domain_nb == 1  :
			stiffness       = apply_dirichlet(V, stiffness, dirichlet = [[False, True],[True, True]])
		else:
			stiffness       = apply_dirichlet(V, stiffness, dirichlet = [[False, False],[True, True]])

		# ...
		M                     = stiffness.tosparse()
		self.M                    = M
		self.lu               = sla.splu(csc_matrix(M))
		self.domain_nb        = domain_nb
		self.S_DDM            = S_DDM
		self.ovlp_value_left  = ovlp_value_left
		self.ovlp_value_right = ovlp_value_right
		self.sp               = [V, V_TOT]
		self.u11_mpH          = u11_mpH 
		self.u12_mpH          = u12_mpH
		self.v11_mpH          = v11_mpH 
		self.v12_mpH          = v12_mpH
		self.w11_mpH          = w11_mpH 
		self.w12_mpH          = w12_mpH
		self.u_d	      = u_d
		self.x_d	   = u_d.toarray().reshape(V.nbasis)
	def solve(self, u_np1, u_np2):   	       
		V, V_TOT            = self.sp[:]
		# ...
		u                   = StencilVector(V.vector_space)
		# ++++
		#--Assembles a right hand side of Poisson equation
		rhs                 = StencilVector(V.vector_space)
		rhs                 = assemble_rhs( V_TOT, fields = [self.u11_mpH, self.u12_mpH, self.u_d, self.v11_mpH, self.v12_mpH, u_np1, self.w11_mpH, self.w12_mpH, u_np2],
									 knots = True, value = [self.domain_nb, self.S_DDM, self.ovlp_value_left, self.ovlp_value_right], out = rhs )
		if self.domain_nb == 0 :
			rhs              = apply_dirichlet(V, rhs, dirichlet = [[True, False],[True, True]])
		elif self.domain_nb == 1:
			rhs              = apply_dirichlet(V, rhs, dirichlet = [[False, True],[True, True]])
		else:
			rhs              = apply_dirichlet(V, rhs, dirichlet = [[False, False],[True, True]])
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
		#res                 =  abs(dot(self.M.toarray()  , u)  -b)
		return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1
NRefine     = 9# please take it as factor of 16 

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

# after refinement
#----------------------------------------
#..... Parameterization from 16*16 elements
#----------------------------------------
# Quart annulus
#geometry  = '../fields/quart_annulus.xml'
# Half annulus
#geometry  = '../parallel_Schwarz_method_Robin_2D/half_annulus.xml'
# Circle
geometry  = '../fields/annulus.xml'
#geometry = '../fields/circle.xml'
# ... Overlape ??
#geometry  = '../fields/Annulus_over1.xml'

print('#---IN-UNIFORM--MESH-Poisson equation patch 1', geometry)

#Annuls : patch 1
# ... Assembling mapping
mp1       = getGeometryMap(geometry,0)
xmp1      = zeros(mp1.nbasis)
ymp1      = zeros(mp1.nbasis)

xmp1[:,:], ymp1[:,:] =  mp1.coefs()

# ... Assembling mapping
mp2             = getGeometryMap(geometry,1) # second part
xmp2 = zeros(mp2.nbasis)
ymp2 = zeros(mp2.nbasis)

xmp2[:,:], ymp2[:,:] =  mp2.coefs()

mp3             = getGeometryMap(geometry,2) # second part
xmp3 = zeros(mp3.nbasis)
ymp3 = zeros(mp3.nbasis)

xmp3[:,:], ymp3[:,:] =  mp3.coefs()
#..-------------- for C0 We take F1 = F1
# ... C1 continuity garad(F1 = F2)  = garad(F2)  interface
xmp2[0,:] = xmp1[-1,:] #C0
ymp2[0,:] = ymp1[-1,:] #C0

xmp3[0,:] = xmp2[-1,:] #C0
ymp3[0,:] = ymp2[-1,:] #C0
print('#--- Poisson equation : ', geometry)

#Annuls : patch 1
# ... Assembling mapping
mp1         = getGeometryMap(geometry,0)
# ... Assembling mapping
mp2         = getGeometryMap(geometry,1) # second part
mp3         = getGeometryMap(geometry, 2)
# ... Refine number of elements
nelements   = (mp1.nelements[0] * NRefine, mp1.nelements[1] * NRefine) #... number of elements

print('Number of elements in each direction : ', nelements)
# ... Refine mapping
xmp1, ymp1  =  mp1.RefineGeometryMap(Nelements= nelements)
xmp2, ymp2  =  mp2.RefineGeometryMap(Nelements= nelements)
xmp3, ymp3  =  mp3.RefineGeometryMap(Nelements= nelements)

#xmp2[1,:] = 2.*xmp1[-1,:] - xmp1[-2,:] #C1
#ymp2[1,:] = 2.*ymp1[-1,:] - ymp1[-2,:]  #C1



#-------------++++++++++++++++-------------------------------------------------
'''
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
plt.plot(phidx, phidy, 'm', linewidth=2., label = 'patch 1 $Im([0,1]^2_{y=0})$')
# ...
phidx = xmp1[:,Vh1.nbasis[1]-1]
phidy = ymp1[:,Vh1.nbasis[1]-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = 'patch 1 $Im([0,1]^2_{y=1})$')
#''

phidx = xmp1[0,:]
phidy = ymp1[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = 'patch 1 $Im([0,1]^2_{x=0})$')
# ...
phidx = xmp1[Vh1.nbasis[1]-1,:]
phidy = ymp1[Vh1.nbasis[1]-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = 'patch 1 $Im([0,1]^2_{x=1}$)')
   
for i in range(Vh1.nbasis[1]):
   phidx = xmp2[:,i]
   phidy = ymp2[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
for i in range(Vh1.nbasis[0]):
   phidx = xmp2[i,:]
   phidy = ymp2[i,:]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
   

for i in range(Vh1.nbasis[1]):
   phidx = xmp3[:,i]
   phidy = ymp3[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
for i in range(Vh1.nbasis[0]):
   phidx = xmp3[i,:]
   phidy = ymp3[i,:]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
   
#.. Plot the surface in the first patch 1

#.. Plot the surface in the second patch 2

phidx = xmp2[:,0]
phidy = ymp2[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = 'patch 2 $Im([0,1]^2_{y=0})$')
# ...
phidx = xmp2[:,Vh1.nbasis[1]-1]
phidy = ymp2[:,Vh1.nbasis[1]-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = 'patch 2 $Im([0,1]^2_{y=1})$')
#''
phidx = xmp2[0,:]
phidy = ymp2[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = 'patch 2 $Im([0,1]^2_{x=0})$')
# ...
phidx = xmp2[Vh1.nbasis[1]-1,:]
phidy = ymp2[Vh1.nbasis[1]-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = 'patch 2 $Im([0,1]^2_{x=1}$)')



phidx = xmp3[:,0]
phidy = ymp3[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = 'patch 3 $Im([0,1]^2_{y=0})$')
# ...
phidx = xmp3[:,Vh1.nbasis[1]-1]
phidy = ymp3[:,Vh1.nbasis[1]-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = 'patch 3 $Im([0,1]^2_{y=1})$')
#''
phidx = xmp3[0,:]
phidy = ymp3[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = 'patch 3 $Im([0,1]^2_{x=0})$')
# ...
phidx = xmp3[Vh1.nbasis[1]-1,:]
phidy = ymp3[Vh1.nbasis[1]-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = 'patch 3 $Im([0,1]^2_{x=1}$)')

plt.legend()
plt.scatter(xmp1[Vh1.nbasis[1]-1,:],ymp1[Vh1.nbasis[1]-1,:], color= 'black', linewidths=3.)
plt.scatter(xmp1[Vh1.nbasis[1]-2,:],ymp1[Vh1.nbasis[1]-2,:],  color= 'red', linewidths=3.)

plt.scatter(xmp2[Vh1.nbasis[1]-1,:],ymp2[Vh1.nbasis[1]-1,:], color= 'black', linewidths=3.)
plt.scatter(xmp2[Vh1.nbasis[1]-2,:],ymp2[Vh1.nbasis[1]-2,:],  color= 'red', linewidths=3.)

plt.text(0.5, -0.5, r'$\Omega_1$', fontsize=20)
plt.text(0.01, -0.5, r'$\Gamma_{12}$', fontsize=20)

plt.text(-0.5, -0.5, r'$\Omega_2$', fontsize=20)
plt.text(-0.5, 0.01, r'$\Gamma_{23}$', fontsize=20)

plt.text(-0.5, 0.5, r'$\Omega_3$', fontsize=20)

plt.show() 

'''
#--------------------------------------------------------------

#...End of parameterisation
#--------------------------------------------------------------
# ... please take into account that : beta < alpha 
alpha_1       = 0.3 # fixed by the geometry parameterization
alpha_2       = 0.6# fixed by the geometry parameterization

overlap     = alpha_2 - alpha_1
xuh_0       = []
xuh_01      = []
xuh_02       = []
iter_max    = 100
tol      = 1e-10
S_DDM       = 1./alpha_1**2 #alpha/(nelements+1)
#.. test 0
u_exact   = lambda x, y : sin(2.*pi*x)*sin(2.*pi*y)
#u_exact   = lambda x, y : 5.0/cosh(50 * ((8*(x-0.5)**2) -y**2* 0.125))*(1.-x**2-y**2)*y
#--------------------------
#..... Initialisation
#--------------------------
grids_0 = linspace(0., alpha_1, nelements[0]+1)
# create the spline space for each direction
V1_0    = SplineSpace(degree=degree, nelements= nelements[0], grid =grids_0, nderiv = 2, quad_degree = quad_degree)
V2_0    = SplineSpace(degree=degree, nelements= nelements[0], nderiv = 2, quad_degree = quad_degree)
V_0     = TensorSpace(V1_0, V2_0)

grids_1 = linspace(alpha_1, alpha_2, nelements[0]+1)
# create the spline space for each direction
V1_1    = SplineSpace(degree=degree, nelements= nelements[0], grid =grids_1, nderiv = 2, quad_degree = quad_degree)
V2_1    = SplineSpace(degree=degree, nelements= nelements[0],  nderiv = 2, quad_degree = quad_degree)
V_1     = TensorSpace(V1_1, V2_1)

grids_2 = linspace(alpha_2, 1., nelements[0]+1)
V1_2    = SplineSpace(degree=degree, nelements= nelements[0], grid =grids_2, nderiv = 2, quad_degree = quad_degree)
V2_2    = SplineSpace(degree=degree, nelements= nelements[0],  nderiv = 2, quad_degree = quad_degree)
V_2    = TensorSpace(V1_2, V2_2)

#...
Vt_0    = TensorSpace(V1_0, V2_0, V1_1, V2_1, V1_2, V2_2 )
Vt_1    = TensorSpace(V1_1, V2_1, V1_0, V2_0, V1_2, V2_2 )
Vt_2    = TensorSpace(V1_2, V2_2, V1_1, V2_1, V1_0, V2_0)

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

w11_mpH        = StencilVector(V_2.vector_space)
w12_mpH        = StencilVector(V_2.vector_space)
w11_mpH.from_array(V_2, xmp3)
w12_mpH.from_array(V_2, ymp3)
# --- Initialization domain in left
nbpts =100
plot_MeshMultipatch(nbpts, (V_0, V_1, V_2), (xmp1, xmp2, xmp3), (ymp1, ymp2, ymp3))
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

P0      = DDM_poisson(V_0, Vt_0, u11_mpH, u12_mpH, v11_mpH, v12_mpH, w11_mpH, w12_mpH, u_d, S_DDM, domain_nb, ovlp_value_left = alpha_1, ovlp_value_right=0. )

# --- Initialization domain in right
domain_nb = 1
sX         = pyccel_sol_field_2d((n_dir,n_dir),  xmp3 , V_2.knots, V_2.degree)[0]
sY         = pyccel_sol_field_2d((n_dir,n_dir),  ymp3 , V_2.knots, V_2.degree)[0]
u_d        = StencilVector(V_2.vector_space)
x_d        = np.zeros(V_2.nbasis)
#x_d[0, : ] = least_square_Bspline(V2_1.degree, V2_1.knots, u_exact(sX[0, :], sY[ 0,:]), m= n_dir)
x_d[-1, :] = least_square_Bspline(V2_1.degree, V2_2.knots, u_exact(sX[-1,:], sY[-1,:]), m= n_dir)
x_d[:,0]   = least_square_Bspline(V1_2.degree, V1_2.knots, u_exact(sX[:, 0], sY[:, 0]), m= n_dir)
x_d[:, -1] = least_square_Bspline(V1_2.degree, V1_2.knots, u_exact(sX[:,-1], sY[:,-1]), m= n_dir)
u_d.from_array(V_2, x_d)
P2      = DDM_poisson(V_2, Vt_2, w11_mpH, w12_mpH, v11_mpH, v12_mpH, u11_mpH, u12_mpH, u_d, S_DDM, domain_nb, ovlp_value_left = alpha_2, ovlp_value_right=1.  )



domain_nb = 2
sX         = pyccel_sol_field_2d((n_dir,n_dir),  xmp2 , V_1.knots, V_1.degree)[0]
sY         = pyccel_sol_field_2d((n_dir,n_dir),  ymp2 , V_1.knots, V_1.degree)[0]
u_d        = StencilVector(V_1.vector_space)
x_d        = np.zeros(V_1.nbasis)
#x_d[0, : ] = least_square_Bspline(V2_1.degree, V2_1.knots, u_exact(sX[0, :], sY[ 0,:]), m= n_dir)
#x_d[-1, :] = least_square_Bspline(V2_1.degree, V2_1.knots, u_exact(sX[-1,:], sY[-1,:]), m= n_dir)
x_d[:,0]   = least_square_Bspline(V1_1.degree, V1_1.knots, u_exact(sX[:, 0], sY[:, 0]), m= n_dir)
x_d[:, -1] = least_square_Bspline(V1_1.degree, V1_1.knots, u_exact(sX[:,-1], sY[:,-1]), m= n_dir)
u_d.from_array(V_1, x_d)
P1      = DDM_poisson(V_1, Vt_1, v11_mpH, v12_mpH, u11_mpH, u12_mpH, w11_mpH, w12_mpH, u_d, S_DDM, domain_nb, ovlp_value_left = alpha_1, ovlp_value_right=alpha_2  )


# ... communication Solution interface
u_00    = StencilVector(V_0.vector_space)
u_11     = StencilVector(V_1.vector_space)
u_2     = StencilVector(V_2.vector_space)

print('#---IN-UNIFORM--MESH')
u_0,   xuh, l2_norm,   H1_norm   = P0.solve(u_11, u_2)
xuh_0.append(xuh)
u_1, xuh_1, l2_norm1, H1_norm1  = P1.solve(u_00, u_2)
xuh_01.append(xuh_1)
u_2, xuh_2, l2_norm2, H1_norm2 = P2.solve(u_11, u_00)
xuh_02.append(xuh_2)
l2_err = l2_norm +l2_norm1 + l2_norm2
H1_err = H1_norm + H1_norm1 + H1_norm2
r = abs(l2_norm-l2_norm1)
u_00 = u_0
u_11 = u_1
print('Iteration {}-----> L^2-error ={} -----> H^1-error = {}-----> Residual =  {}'.format(0,f"{l2_err:.2e}",  f"{H1_err:.2e}", f"{r:.2e}" ))

for i in range(iter_max):
	u_0,   xuh, l2_norm, H1_norm    = P0.solve(u_11, u_2)
	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1   = P1.solve(u_00, u_2)
	xuh_01.append(xuh_1)
	u_2, xuh_2, l2_norm2, H1_norm2   = P2.solve(u_11, u_00)
	xuh_02.append(xuh_2)
	if r <=tol :
		iter_max = i+1
		print('-------------------------------------------------------------------------------------')
		print('\t\t the tolerance', tol, 'is reached at the', iter_max,'th iteration')
		print('---------------------------------------------------------------------------------------')
		break
	l2_err = l2_norm + l2_norm1 +l2_norm2
	H1_err = H1_norm + H1_norm1 + H1_norm2
	r = abs(l2_norm-l2_norm1)
	u_00 = u_0
	u_11 = u_1
	print('')
	print('Iteration {}-----> L^2-error ={} -----> H^1-error = {}-----> Residual =  {}'.format(i+1, f"{l2_err:.2e}",  f"{H1_err:.2e}", f"{r:.2e}" ))

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
	
	u_2, ux_2, uy_2, X_2, Y_2     = pyccel_sol_field_2d((nbpts, nbpts),  xuh_2,   V_2.knots, V_2.degree)
	u_0  = []
	u_01 = []
	u_02 = []
	for i in range(iter_max):
		u_0.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_0[i],   V_0.knots, V_0.degree)[0][:,50])
		u_01.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_01[i],   V_1.knots, V_1.degree)[0][:,50])
		u_02.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_02[i],   V_2.knots, V_2.degree)[0][:,50])
	plt.figure() 
	plt.axes().set_aspect('equal')
	plt.subplot(121)
	for i in range(iter_max-1):
		plt.plot(X[:,50], u_0[i], '-k', linewidth = 1.)
		plt.plot(X_1[:,50], u_01[i], '-k', linewidth = 1.)
		plt.plot(X_2[:,50], u_02[i], '-k', linewidth = 1.)  
	plt.plot(X[:,50], u_0[i+1], '-k', linewidth = 1., label='$\mathbf{Un_0-iter(i)}$')
	plt.plot(X_1[:,50], u_01[i+1], '-k', linewidth = 1., label='$\mathbf{Un_1-iter(i)}$')
	plt.plot(X_2[:,50], u_02[i+1], '-k', linewidth = 1., label='$\mathbf{Un_2-iter(i)}$')
	plt.legend()
	plt.grid(True)  
	plt.subplot(122)
	plt.plot(X[:,50], u[:,50],  '--or', label = '$\mathbf{Un_0-iter-max}$' )
	plt.plot(X_1[:,50], u_1[:,50],  '--om', label = '$\mathbf{Un_1-iter-max}$')
	plt.plot(X_2[:,50], u_2[:,50],  '--p', label = '$\mathbf{Un_2-iter-max}$')

	plt.legend()
	plt.grid(True)  
	plt.savefig('behvoir_between_two_patches.png')
	plt.show()

	
	u1, ux, uy, X, Y = pyccel_sol_field_2d((nbpts, nbpts), xuh, V_0.knots, V_0.degree)
	F1_1 = pyccel_sol_field_2d((nbpts, nbpts), xmp1, V_0.knots, V_0.degree)[0]
	F2_1 = pyccel_sol_field_2d((nbpts, nbpts), ymp1, V_0.knots, V_0.degree)[0]

	# --- Second Subdomain ---
	u2, ux, uy, X, Y = pyccel_sol_field_2d((nbpts, nbpts), xuh_1, V_1.knots, V_1.degree)
	F1_2 = pyccel_sol_field_2d((nbpts, nbpts), xmp2, V_1.knots, V_1.degree)[0]
	F2_2 = pyccel_sol_field_2d((nbpts, nbpts), ymp2, V_1.knots, V_1.degree)[0]
	
	u3, ux, uy, X, Y = pyccel_sol_field_2d((nbpts, nbpts), xuh_2, V_2.knots, V_2.degree)
	F1_3 = pyccel_sol_field_2d((nbpts, nbpts), xmp3, V_2.knots, V_2.degree)[0]
	F2_3 = pyccel_sol_field_2d((nbpts, nbpts), ymp3, V_2.knots, V_2.degree)[0]

	# --- Compute Global Color Levels ---
	u_min = min(np.min(u1), np.min(u2),  np.min(u3))
	u_max = max(np.max(u1), np.max(u2), np.max(u3))
	levels = np.linspace(u_min, u_max, 100)  # Uniform levels for both plots

	# --- Create Figure ---
	fig, axes = plt.subplots(figsize=(8, 6))

	# --- Contour Plot for First Subdomain ---
	im1 = axes.contourf(F1_1, F2_1, u1, levels, cmap='jet')

	# --- Contour Plot for Second Subdomain ---
	im2 = axes.contourf(F1_2, F2_2, u2, levels, cmap='jet')
	
	im3 = axes.contourf(F1_3, F2_3, u3, levels, cmap='jet')

	# --- Colorbar ---
	divider = make_axes_locatable(axes)
	cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
	cbar = plt.colorbar(im3, cax=cax)
	cbar.ax.tick_params(labelsize=15)
	cbar.ax.yaxis.label.set_fontweight('bold')

	# --- Formatting ---
	axes.set_title("Solution the in whole domain ", fontweight='bold')
	for label in axes.get_xticklabels() + axes.get_yticklabels():
		label.set_fontweight('bold')

	fig.tight_layout()
	plt.savefig('2patch.png')
	plt.show()

