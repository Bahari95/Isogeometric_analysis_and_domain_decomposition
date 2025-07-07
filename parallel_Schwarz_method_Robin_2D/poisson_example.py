from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import getGeometryMap

from simplines import build_dirichlet
#---In Poisson equation
from gallery_section_04 import assemble_vector_un_ex01   
from gallery_section_04 import assemble_matrix_un_ex01 
from gallery_section_04 import assemble_norm_ex01     

assemble_stiffness2D = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#..
from   core.plot                    import plotddm_result
from   simplines                    import plot_SolutionMultipatch, plot_JacobianMultipatch
from   simplines                    import plot_MeshMultipatch

from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray, linspace, tanh
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2, cosh
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

degree      = 2  # fixed by parameterization for now
quad_degree = degree + 1
NRefine     = 32# nelements refined NRefine times 

#---------------------------------------- 
#..... Geometry parameterization
#----------------------------------------
#.. test 0
#g         = ['sin(2.*pi*x)*sin(2.*pi*y)']
#.. test 1
g        = ['tanh( (0.4-np.sqrt((x-0.5)**2+ (y-0.5)**2))/(sqrt(2)*0.07))']

#----------------------------------------
#..... Parameterization from 16*16 elements
#----------------------------------------
# Quart annulus
geometry  = '../fields/quart_annulus.xml'
# Half annulus
#geometry  = '../fields/annulus.xml'
# Circle
#geometry = '../fields/circle.xml'
# Lshape
#geometry  = '../fields/lshape.xml'
# DDM shape
#geometry  = '../fields/ddm.xml'
# DDM shape
#geometry  = '../fields/ddm2.xml'
# DDM shape
#geometry  = '../fields/ddm3.xml'
# ... Overlape ??
#geometry  = '../fields/Annulus_over1.xml'

#geometry  = '../fields/annulus_48.xml'

#geometry  = '../fields/annulus_G1.xml'

print('#--- Poisson equation : ', geometry)

#Annuls : patch 1
# ... Assembling mapping
mp1         = getGeometryMap(geometry,0)
# ... Assembling mapping
mp2         = getGeometryMap(geometry,1) # second part

# ... Refine number of elements
nelements   = (mp1.nelements[0] * NRefine, mp1.nelements[1] * NRefine) #... number of elements

print('Number of elements in each direction : ', nelements)
# ... Refine mapping
xmp1, ymp1  =  mp1.RefineGeometryMap(Nelements= nelements)
xmp2, ymp2  =  mp2.RefineGeometryMap(Nelements= nelements)

#xmp2[1,:] = xmp2[1,:]+ 0.001


#--------------------------------------------------------------
#...End of parameterisation
#--------------------------------------------------------------
# ... please take into account that : beta < alpha 
nbpts       = 100 # number of points for plot
alpha       = 1. # fixed by the geometry parameterization
beta        = 1. # fixed by the geometry parameterization
iter_max    = 100
tol         = 1e-10
L           = 1  #the size of the parametric space
S_DDM       = (pi/(2*L)) #lowest eigein values
xuh_0       = []
xuh_01      = []
u_exact     = lambda x, y : eval(g[0])
#--------------------------
#..... Initialisation
#--------------------------
grids_0 = linspace(0, alpha, nelements[0]+1)
print( nelements[0])
# create the spline space for each direction
V1_0    = SplineSpace(degree=degree, nelements= nelements[0], grid = grids_0, nderiv = 2, quad_degree = quad_degree)
V2_0    = SplineSpace(degree=degree, nelements= nelements[1], nderiv = 2, quad_degree = quad_degree)
V_0     = TensorSpace(V1_0, V2_0)

grids_1 = linspace(beta, 2., nelements[0]+1)
# create the spline space for each direction
V1_1    = SplineSpace(degree=degree, nelements= nelements[0], grid = grids_1, nderiv = 2, quad_degree = quad_degree)
V2_1    = SplineSpace(degree=degree, nelements= nelements[1],  nderiv = 2, quad_degree = quad_degree)
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
#.. Dirichlet boundary condition
x_d, u_d = build_dirichlet(V_0, g, map = (xmp1, ymp1))
x_d[-1, :] = 0.
u_d.from_array(V_0, x_d)
P0      = DDM_poisson(V_0, Vt_0, u11_mpH, u12_mpH, v11_mpH, v12_mpH, u_d, S_DDM, domain_nb, alpha )
#for S_DDM     in np.arange(0.1, 100, 0.1):
# --- Initialization domain in right
domain_nb = 1
#.. Dirichlet boundary condition
x_d,  u_d = build_dirichlet(V_1, g, map = (xmp2, ymp2))

x_d[0, :] = 0.
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
r = abs(l2_norm -l2_norm1)
print('Iteration {}-----> L^2-error ={} -----> H^1-error = {}-----> Residual =  {}'.format(0,f"{l2_err:.2e}",  f"{H1_err:.2e}", f"{r:.2e}" ))

for i in range(iter_max):
	#...
	u_0, xuh, l2_norm, H1_norm     = P0.solve(u_1)

	xuh_0.append(xuh)
	u_1, xuh_1, l2_norm1, H1_norm1 = P1.solve(u_00)
	
	xuh_01.append(xuh_1)
	u_00   = u_0
	if r <=tol :
		iter_max = i+1
		print('-------------------------------------------------------------------------------------')
		print('\t\t the tolerance', tol, 'is reached at the', iter_max,'th iteration')
		print('---------------------------------------------------------------------------------------')
		break
	l2_err = l2_norm + l2_norm1
	H1_err = H1_norm + H1_norm1
	r = abs(l2_norm -l2_norm1)
	print('')
	print('Iteration {}-----> L^2-error ={} -----> H^1-error = {}-----> Residual =  {}'.format(i+1, f"{l2_err:.2e}",  f"{H1_err:.2e}", f"{r:.2e}" ))
	print('')
	print('')
	#---Compute a solution
plotddm_result(nbpts, (xuh_0,  xuh_01), (V_0, V_1), (xmp1, xmp2))

from simplines import paraview_SolutionMultipatch

paraview_SolutionMultipatch(nbpts, (V_0, V_1), (xmp1, xmp2), (ymp1, ymp2), xuh = (xuh,  xuh_1), Func = u_exact)
# plot_JacobianMultipatch(nbpts, (V_0, V_1), (xmp1, xmp2), (ymp1, ymp2))
# plot_MeshMultipatch(nbpts, (V_0, V_1), (xmp1, xmp2), (ymp1, ymp2))
