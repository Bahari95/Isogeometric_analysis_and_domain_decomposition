from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import plot_field_1d
from simplines import least_square_Bspline
from simplines import point_on_bspline_curve
#.. Prologation by knots insertion matrix
from   simplines                    import prolongation_matrix
import time
start = time.time()

#---In Poisson equation
from d1_gallery_section_04 import assemble_vector_ex01
from d1_gallery_section_04 import assemble_vector_ex11

from d1_gallery_section_04 import assemble_massmatrix1D
from d1_gallery_section_04 import assemble_matrix_un_ex01
from d1_gallery_section_04 import assemble_norm_ex01 


assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

assemble_stiffness   = compile_kernel(assemble_matrix_un_ex01, arity=2)

assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_rhsc        = compile_kernel(assemble_vector_ex11, arity=1)

assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)
print('time to import utilities of Poisson equation =', time.time()-start)

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

class DDM_Helmholtz(object):

    def __init__(self, V, domain_nb, Kappa, ovlp_value):

       # ...
       stiffness  = assemble_stiffness(V, value = [Kappa])

       if domain_nb == 0 :
            #-- Build lift hand side of a linear system
            n_basis                    = V.nbasis-2
            M                          = zeros((n_basis*2,n_basis*2))
            # ..
            b                          = zeros(n_basis*2)
            #M[0,0]                     = -Kappa
            #M[n_basis+1,n_basis+1]     =  Kappa
            M[n_basis:,:n_basis]       = (stiffness.tosparse()).toarray()[1:-1,1:-1]
            M[:n_basis,n_basis:]       = (stiffness.tosparse()).toarray()[1:-1,1:-1]

       else :
            n_basis                    = V.nbasis-1
            M                          = zeros((n_basis*2,n_basis*2))
            # ..
            b                          = zeros(n_basis*2)
            M[n_basis-1,n_basis-1]     = -Kappa
            M[-1,-1]                   =  Kappa
            M[n_basis:,:n_basis]       = (stiffness.tosparse()).toarray()[1:,1:]
            M[:n_basis,n_basis:]       = (stiffness.tosparse()).toarray()[1:,1:]

       # ...
       self.lu             = sla.splu(csc_matrix(M))
       # ...
       self.V              = V
       self.ovlp_value     = ovlp_value
       self.domain_nb      = domain_nb
       self.n_basis        = n_basis
       self.Kappa          = Kappa
                     
    def solve(self,  x_d, x_dc):
       '''
       Solve Helmholtz equation
       '''
       u_d                   = StencilVector(self.V.vector_space)
       u_dc                  = StencilVector(self.V.vector_space)
       if self.domain_nb == 0 :
	       x_dh                  = zeros(self.V.nbasis)
	       x_dh[-1] = x_d   #np.cos(self.Kappa*self.ovlp_value) # x_d
	       x_dh[0]  =  1.
	       u_d.from_array(self.V, x_dh)
	       x_dhc                  = zeros(self.V.nbasis)
	       x_dhc[-1] = x_dc #  np.sin(self.Kappa*self.ovlp_value) #x_dc
	       u_dc.from_array(self.V, x_dhc)
       else :
	       x_dh                  = zeros(self.V.nbasis)
	       x_dh[0]  = x_d  #np.cos(self.Kappa*self.ovlp_value)
	       u_d.from_array(self.V, x_dh)
	       x_dhc                  = zeros(self.V.nbasis)
	       x_dhc[0] = x_dc #np.sin(self.Kappa*self.ovlp_value)
	       u_dc.from_array(self.V, x_dhc)
       u                     = StencilVector(self.V.vector_space)
       v                     = StencilVector(self.V.vector_space)
       # ... assembles rhs 
       rhs                   = assemble_rhs( self.V, fields = [u_d], value = [self.Kappa])
       rhsc                  = assemble_rhsc(self.V, fields = [u_dc], value = [self.Kappa] )

       #-- Build right hand side of a linear system
       b                     = zeros(self.n_basis*2)
       if self.domain_nb == 0 :
          b[:self.n_basis]   = rhsc.toarray()[1:-1] 
          b[self.n_basis:]   = rhs.toarray()[1:-1]
       else :
          #print(rhs.toarray()[-1], rhsc.toarray()[-1])
          b[:self.n_basis]   = rhsc.toarray()[1:]
          b[self.n_basis:]   = rhs.toarray()[1:]

       x                     = self.lu.solve(b)
       #... Dirichlet nboundary
       if self.domain_nb == 0 :
	       xr                    = [1.] + list(x[:self.n_basis]) + [x_d]
	       u.from_array(self.V, xr)
	       # ...       
	       xc                    = [0.] + list(x[self.n_basis:]) + [x_dc]
	       v.from_array(self.V, xc)
       else :
	       xr                    = [x_d] + list(x[:self.n_basis])
	       u.from_array(self.V, xr)
	       # ...       
	       xc                    = [x_dc] + list(x[self.n_basis:])
	       v.from_array(self.V, xc)
       Norm                  = assemble_norm_l2(self.V, fields=[u], value = [0, self.Kappa])
       norm                  = Norm.toarray()
       l2_norm               = norm[0]
       H1_norm               = norm[1]

       Norm                  = assemble_norm_l2(self.V, fields=[v], value = [1, self.Kappa])
       norm                  = Norm.toarray()
       l2_normc              = norm[0]
       H1_normc              = norm[1]
       
       return u, v, xr, xc, l2_norm, H1_norm, l2_normc, H1_normc    
    

degree      = 2
quad_degree = degree + 1

# ... please take into account that : beta < alpha 
alpha       = 0.7
beta        = 0.2
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 1

Kappa       = 20. 
#----------------------
#..... Initialisation 
#----------------------
nelements  = 128

grids_0 = linspace(0, alpha, nelements+1)
# create the spline space for each direction
V_0    = SplineSpace(degree=degree, nelements= nelements, grid =grids_0, nderiv = 2, quad_degree = quad_degree)

grids_1 = linspace(beta, 1., nelements+1)
# create the spline space for each direction
V_1    = SplineSpace(degree=degree, nelements= nelements, grid =grids_1, nderiv = 2, quad_degree = quad_degree)

# ... INITIALIZATION
H0      = DDM_Helmholtz(V_0, 0, Kappa, alpha)
H1      = DDM_Helmholtz(V_1, 1, Kappa, beta)

print('#---IN-UNIFORM--MESH')
u_0, u_0c, xuh, xuhc, l2_norm, H1_norm, l2_normc, H1_normc  = H0.solve( 0., 0.)
xuh_0.append(xuh)
u_1, u_1c, xuh_1, xuh_1c, l2_norm1, H1_norm1, l2_norm1c, H1_norm1c  = H1.solve(0., 0.)
xuh_01.append(xuh_1)

l2_err = l2_norm + l2_norm1
H1_err = H1_norm + H1_norm1

print('0r-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm, H1_norm))
print('0c-----> L^2-error ={} -----> H^1-error = {}'.format(l2_normc, H1_normc))

print('1r-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm1, H1_norm1))
print('1c-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm1c, H1_norm1c))
P = np.zeros((V_0.nbasis, 1))

plt.figure()
for i in range(iter_max):
	# ... computes the image of the overlape point by a new solution
	P[:,0]  = xuh_1[:]
	a = point_on_bspline_curve(V_1.knots, P, alpha)
	P[:,0]  = xuh_1c[:]
	b = point_on_bspline_curve(V_1.knots, P, alpha)
	u_0, u_0c, xuh, xuhc, l2_norm, H1_norm, l2_normc, H1_normc  = H0.solve( a, b)
	xuh_0.append(xuh)
	P[:,0]  = xuh[:]
	a = point_on_bspline_curve(V_0.knots, P, beta)
	P[:,0]  = xuhc[:]
	b = point_on_bspline_curve(V_0.knots, P, beta)
	u_1, u_1c, xuh_1, xuh_1c, l2_norm1, H1_norm1, l2_norm1c, H1_norm1c  = H1.solve( a, b)
	xuh_01.append(xuh_1)
	#if abs(l2_err - l2_norm - l2_norm1) <=1e-8:
	#        iter_max = i+1
	#        print(iter_max)
	#        break
	l2_err = sqrt(l2_norm**2 + l2_norm1**2)
	H1_err = sqrt(H1_norm**2 + H1_norm1**2)
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))
	#---Compute a solution
	if i%20 == 0:
		plot_field_1d(V_0.knots, V_0.degree, xuh, color = None)
		# ...
		plot_field_1d(V_1.knots, V_1.degree,  xuh_1, color = None)#, label = 'ERR1_{%d}_{%f}' % (i, l2_norm1))
	
plt.legend(fontsize="15")
plt.savefig('Helmholtz.png')
plt.show()
#---Compute a solution
nbpts = 100
# # ........................................................
# ....................For a plot
# #.........................................................
if True :
	plt.figure()
	#---Compute a solution
	plot_field_1d(V_0.knots, V_0.degree, xuh, color = None)
	# ...
	plot_field_1d(V_1.knots, V_1.degree,  xuh_1, color = None)
	plt.savefig('Helmholtz_R.png')
	plt.show()
	plt.figure()
	#.. Plot the densities
	plot_field_1d(V_0.knots, V_0.degree, xuhc, color = None)
	# ...
	plot_field_1d(V_1.knots, V_1.degree, xuh_1c, color = None)
	plt.savefig('Helmholtz_C.png')
	plt.show()
