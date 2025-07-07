from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import getGeometryMap

from simplines import build_dirichlet
#---In Poisson equation

from gallery_section_04_GG import assemble_matrix_un_ex01 
from gallery_section_04_GG import assemble_norm_ex01 
from gallery_section_04_GG import assemble_vector_un_ex01_without_ddm    

assemble_stiffness2D = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_GEOM    = compile_kernel(assemble_vector_un_ex01_without_ddm, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#..
from   core.plot                    import plotddm_result, plot_Solution
from   core.plot                    import plot_Mesh, plot_surface_solution

from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray, linspace
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2, cosh
from   tabulate                     import tabulate
import numpy                        as     np
import timeit
import time

#==============================================================================
#.......Poisson ALGORITHM
#==============================================================================		

class GENARAL(object):
	def __init__(self, V,  u11_mpH, u12_mpH):
		# ++++
		stiffness           = assemble_stiffness2D(V, fields = [u11_mpH, u12_mpH] )
	
		stiffness          = apply_dirichlet(V, stiffness, dirichlet = [[True, False],[True, True]])
		
		# ...
		M                  = stiffness.tosparse()
	
		self.sharing       = [sla.splu(csc_matrix(M)), V, u11_mpH,  u12_mpH]
		
		
	def solve(self, u_d):   	       
		lu,  V            = self.sharing[:2]  
		u11_mpH, u12_mpH  = self.sharing[2:] 
		# ...
		u                 = StencilVector(V.vector_space)
		# ++++
		#--Assembles a right hand side of Poisson equation
		rhs               = StencilVector(V.vector_space)
		rhs               = assemble_rhs_GEOM( V, fields = [u11_mpH, u12_mpH, u_d], out = rhs )
		
		rhs               = apply_dirichlet(V, rhs, dirichlet = [[True ,False],[True, True]])
		x_d		   = u_d.toarray().reshape(V.nbasis)
		b                 = rhs.toarray()
		# ...
		x                 = lu.solve(b)		# ...
		x                 = x.reshape(V.nbasis) + x_d
		#...
		u.from_array(V, x)
		# ...
		Norm                = assemble_norm_l2(V, fields=[u11_mpH, u12_mpH, u]) 
		norm                = Norm.toarray()
		l2_norm             = norm[0]
		H1_norm             = norm[1]       
		return u, x, l2_norm, H1_norm
        
                

degree      = 2  # fixed by parameterization for now
quad_degree = degree + 1
NRefine     = 2# nelements refined NRefine times 

#---------------------------------------- 
#..... Geometry parameterization
#----------------------------------------
#.. test 0
#g         = ['x*sin(pi*y)'] #Dirichlet at x = 1  # non vanishing and all Dirichelt homo
#.. test 1
g         = [' x**2*y*(1 - y)'] #Neumann at x =1 # non vanishing and all Dirichelt homo

#g         = ['sin(pi*y)*sin(pi*x)']
#----------------------------------------
#..... Parameterization from 16*16 elements
#----------------------------------------
# Quart annulus

#geometry  = 'fields/quart_annulus.xml'
# Half annulus

geometry  = 'fields/annulus.xml'
# Circle
#geometry = 'fields/circle.xml'


print('#--- Poisson equation : ', geometry)

#Annuls : patch 1
# ... Assembling mapping
mp1         = getGeometryMap(geometry,0)


# ... Refine number of elements
nelements   = (mp1.nelements[0] * NRefine, mp1.nelements[1] * NRefine) #... number of elements

print('Number of elements in each direction : ', nelements)
# ... Refine mapping
xmp1, ymp1  =  mp1.RefineGeometryMap(Nelements= nelements)




#--------------------------------------------------------------
#...End of parameterisation
#--------------------------------------------------------------

nbpts       = 100 # number of points for plot
u_exact     = lambda x, y : eval(g[0])
#--------------------------
#..... Initialisation

# create the spline space for each direction
V1_0    = SplineSpace(degree=degree, nelements= nelements[0], nderiv = 2, quad_degree = quad_degree)
V2_0    = SplineSpace(degree=degree, nelements= nelements[1], nderiv = 2, quad_degree = quad_degree)
V_0     = TensorSpace(V1_0, V2_0)
u11_mpH        = StencilVector(V_0.vector_space)
u12_mpH        = StencilVector(V_0.vector_space)
u11_mpH.from_array(V_0, xmp1)
u12_mpH.from_array(V_0, ymp1)
x_d, u_d = build_dirichlet(V_0, g, map = (xmp1, ymp1))

u_d.from_array(V_0, x_d)

P1      = GENARAL(V_0, u11_mpH, u12_mpH )

u, x, l2_norm, H1_norm = P1.solve(u_d)


print('L^2-error ={} -----> H^1-error = -------------->{}'.format(f"{l2_norm:.2e}",  f"{H1_norm:.2e}" ))
plot_Solution(nbpts, x, V_0, xmp1, ymp1)
plot_surface_solution(nbpts, x, V_0, xmp1, ymp1, U_ex = u_exact)
plot_Mesh(nbpts, V_0,  xmp1, ymp1)

