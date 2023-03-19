from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import least_square_Bspline

#.. Prologation by knots insertion matrix
from   simplines                    import prolongation_matrix
import time
start = time.time()

#---In Poisson equation
from gallery_section_04 import assemble_vector_ex01 #---1 : In uniform mesh
from gallery_section_04 import assemble_vector_ex11 #---1 : In uniform mesh
from gallery_section_04 import assemble_vector_ex02

from gallery_section_04 import assemble_massmatrix1D
from gallery_section_04 import assemble_matrix_un_ex01 #---1 : In uniform mesh
from gallery_section_04 import assemble_matrix_un_ex11 #---1 : In uniform mesh

from gallery_section_04 import assemble_norm_ex01 #---1 : In uniform mesh


assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

assemble_stiffness   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_stiffnessc  = compile_kernel(assemble_matrix_un_ex11, arity=2)

assemble_Pr          = compile_kernel(assemble_vector_ex02, arity=1)
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
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray, linspace
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   tabulate                     import tabulate
import numpy                        as     np

#==============================================================================       
def   Pr_h_solve(V1, V2, V, Vt, u, domain_nb, ovlp_value): 

       # Stiffness and Mass matrix in 1D in the first deriction
       M1         = assemble_mass1D(V1)
       M1         = M1.tosparse()
       # ...
       M          = M1
       lu         = sla.splu(csc_matrix(M))

       #...
       rhs        = StencilVector(V1.vector_space)
       rhs        = assemble_Pr(Vt, fields = [u], knots = True, value = [ovlp_value], out = rhs) 
       b          = rhs.toarray()
       
       #---Solve a linear system
       x          = lu.solve(b)
       # ---
       x_n        = zeros(V.nbasis)
       if domain_nb == 0 :
          x_n[-1,:]  = x[:]
       else :
          x_n[0,:]   = x[:]       
       # ---
       u_L2       = StencilVector(V.vector_space)
       u_L2.from_array(V, x_n)
       
       return u_L2, x_n

#==============================================================================
#.......Poisson ALGORITHM
def Helmholtz_solve(V1, V2 , V, domain_nb, Vh = None, u_d = None, u_dc = None):

       u   = StencilVector(V.vector_space)
       v   = StencilVector(V.vector_space)
       # ...
       stiffness  = assemble_stiffness(V)
       stiffnessc = assemble_stiffnessc(V)

       rhs        = assemble_rhs( V, fields = [u_dc])
       rhsc       = assemble_rhsc( V, fields = [u_d] )

       #--Solve a linear system
       n_basis                    = V1.nbasis*V2.nbasis
       M                          = zeros((n_basis*2,n_basis*2))
       # ..
       b                          = zeros(n_basis*2)
       if domain_nb == 0 :
          stiffness  = apply_dirichlet(V, stiffness, dirichlet =[[False, True],[False, False]])
          rhs        = apply_dirichlet(V, rhs, dirichlet =[[False, True],[False, False]])

          stiffnessc = apply_dirichlet(V, stiffnessc, dirichlet =[[False, True],[False, False]])
          rhsc       = apply_dirichlet(V, rhsc, dirichlet =[[False, True],[False, False]])
          
          M[:n_basis,:n_basis]       = -1*(stiffnessc.tosparse()).toarray()[:,:]
          M[n_basis:,n_basis:]       =    (stiffnessc.tosparse()).toarray()[:,:]
          M[n_basis:,:n_basis]       = (stiffness.tosparse()).toarray()[:,:]
          M[:n_basis,n_basis:]       = (stiffness.tosparse()).toarray()[:,:]

          b[:n_basis]                = rhs.toarray()[:] 
          b[n_basis:]                = rhsc.toarray()[:]
       else :
          stiffness  = apply_dirichlet(V, stiffness, dirichlet =[[True, False],[False, False]])
          rhs        = apply_dirichlet(V, rhs, dirichlet =[[True, False],[False, False]])

          stiffnessc = apply_dirichlet(V, stiffnessc, dirichlet =[[True, False],[False, False]])
          rhsc       = apply_dirichlet(V, rhsc, dirichlet =[[True, False],[False, False]])
          
          M[:n_basis,:n_basis]       = -1*(stiffnessc.tosparse()).toarray()[:,:]
          M[n_basis:,n_basis:]       =    (stiffnessc.tosparse()).toarray()[:,:]
          M[n_basis:,:n_basis]       = (stiffness.tosparse()).toarray()[:,:]
          M[:n_basis,n_basis:]       = (stiffness.tosparse()).toarray()[:,:]

          b[:n_basis]                = rhs.toarray()[:] 
          b[n_basis:]                = rhsc.toarray()[:]
       #cond_M = linalg.cond(M.toarray())
       lu      = sla.splu(csc_matrix(M))

       x       = lu.solve(b)
       #... Dirichlet nboundary
       xr       = x[:n_basis].reshape(V.nbasis)
       xr[0,:  ]            += u_d.toarray().reshape(V.nbasis)[0,:]
       xr[-1,: ]            += u_d.toarray().reshape(V.nbasis)[-1,:]
       u.from_array(V, xr)
       
       xc       = x[n_basis:].reshape(V.nbasis)
       xc[0,:  ]            += u_dc.toarray().reshape(V.nbasis)[0,:]
       xc[-1,: ]            += u_dc.toarray().reshape(V.nbasis)[-1,:]
       v.from_array(V, xc)
       
       Norm          = assemble_norm_l2(V, fields=[u], value = [0])
       norm          = Norm.toarray()
       l2_norm       = norm[0]
       H1_norm       = norm[1]

       Norm          = assemble_norm_l2(V, fields=[v], value = [1])
       norm          = Norm.toarray()
       l2_normc       = norm[0]
       H1_normc       =  norm[1]
       
       return u, v, xr, xc, l2_norm, H1_norm, l2_normc, H1_normc
    

degree      = 4
quad_degree = degree + 1

# ... please take into account that : beta < alpha 
alpha       = 0.75
beta        = 0.25
overlap     = alpha - beta
xuh_0       = []
xuh_01      = []
iter_max    = 60

#----------------------
#..... Initialisation and computing optimal mapping for 16*16
#----------------------
nelements  = 16

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
Vt_0    = TensorSpace(V2_0, V1_1, V2_1)
Vt_1    = TensorSpace(V2_1, V1_0, V2_0)
# ... communication Dirichlet interface
uh_d1   = StencilVector(V_0.vector_space)
uh_d1c   = StencilVector(V_0.vector_space)
uh_d0   = StencilVector(V_1.vector_space)
uh_d0c   = StencilVector(V_1.vector_space)

print('#---IN-UNIFORM--MESH')
u_0, u_0c, xuh, xuhc, l2_norm, H1_norm, l2_normc, H1_normc  = Helmholtz_solve(V1_0, V2_0, V_0, 0, u_d= uh_d1, u_dc= uh_d1c)
xuh_0.append(xuh)
u_1, u_1c, xuh_1, xuh_1c, l2_norm1, H1_norm1, l2_norm1c, H1_norm1c  = Helmholtz_solve(V1_1, V2_1, V_1, 1, u_d= uh_d0, u_dc= uh_d0c)
xuh_01.append(xuh_1)

l2_err = l2_norm + l2_norm1
H1_err = H1_norm + H1_norm1

print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

for i in range(iter_max):
	# ... Dirichlezt boudndary condition in x = 0.75 and 0.25
	uh_d1, xh_d                      = Pr_h_solve(V2_0, V2_1, V_0, Vt_0, u_1, 0, alpha)
	uh_d1c, xh_dc                    = Pr_h_solve(V2_0, V2_1, V_0, Vt_0, u_1c, 0, alpha)
	uh_d0, xh                        = Pr_h_solve(V2_1, V2_0, V_1, Vt_1, u_0, 1, beta)
	uh_d0c, xhc                      = Pr_h_solve(V2_1, V2_0, V_1, Vt_1, u_0c, 1, beta)
	#...
	u_0, u_0c, xuh, xuhc, l2_norm, H1_norm, l2_normc, H1_normc  = Helmholtz_solve(V1_0, V2_0, V_0, 0, u_d= uh_d1, u_dc= uh_d1c)
	xuh_0.append(xuh)
	u_1, u_1c, xuh_1, xuh_1c, l2_norm1, H1_norm1, l2_norm1c, H1_norm1c  = Helmholtz_solve(V1_1, V2_1, V_1, 1, u_d= uh_d0, u_dc= uh_d0c)
	xuh_01.append(xuh_1)
	l2_err = l2_norm + l2_norm1
	H1_err = H1_norm + H1_norm1
	print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_err, H1_err))

#---Compute a solution
nbpts = 100
# # ........................................................
# ....................For testing in one nelements
# #.........................................................
if True :
	#---Compute a solution
	u, ux, uy, X, Y               = pyccel_sol_field_2d((nbpts, nbpts),  xuh,   V_0.knots, V_0.degree)
	# ...
	u_1, ux_1, uy_1, X_1, Y_1     = pyccel_sol_field_2d((nbpts, nbpts),  xuh_1,   V_1.knots, V_1.degree)
	#.. Plot the densities
	uc                            = pyccel_sol_field_2d((nbpts, nbpts),  xuhc,   V_0.knots, V_0.degree)[0]
	# ...
	u_c1                          = pyccel_sol_field_2d((nbpts, nbpts),  xuh_1c,   V_1.knots, V_1.degree)[0]
	figtitle  = 'real and comlex part of the solution '

	fig, axes = plt.subplots( 1, 2, figsize=[12,12], gridspec_kw={'width_ratios': [2.5, 2.5]} , num=figtitle )
	for ax in axes:
	   ax.set_aspect('equal')

	ima    = axes[0].contourf( X, Y, u, cmap= 'jet')
	ima    = axes[0].contourf( X_1, Y_1, u_1, cmap= 'jet')
	divider = make_axes_locatable(axes[0]) 
	cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
	plt.colorbar(ima, cax=cax)

	im    = axes[1].contourf( X, Y, uc, cmap= 'jet')
	im    = axes[1].contourf( X_1, Y_1, u_c1, cmap= 'jet')
	divider = make_axes_locatable(axes[1]) 
	cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
	plt.colorbar(im, cax=cax)
	fig.tight_layout()
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('real_complex_Helmholtz.png')
	plt.show()
	u_0  = []
	u_01 = []
	for i in range(iter_max):
	    u_0.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_0[i],   V_0.knots, V_0.degree)[0][:,50])
	    u_01.append(pyccel_sol_field_2d((nbpts, nbpts),  xuh_01[i],   V_1.knots, V_1.degree)[0][:,50])
	# ...
	#solut = lambda x, t, y : sin(pi*t)*x**2*y*3*sin(4.*pi*(1.-x))*(1.-y) 
	solut = lambda  x, y :  cos(20.*(x*cos(pi/4.) + y*sin(pi/4.)))

	plt.figure() 
	plt.axes().set_aspect('equal')
	plt.subplot(121)
	for i in range(iter_max-1):
	     plt.plot(X[:,50], u_0[i], '-k', linewidth = 1.)
	     plt.plot(X_1[:,50], u_01[i], '-k', linewidth = 1.)
	plt.plot(X_1[:,50], u_01[i+1], '-k', linewidth = 1., label='$\mathbf{Un_1-iter(i)}$')
	plt.plot(X[:,50], u_0[i+1], '-k', linewidth = 1., label='$\mathbf{Un_0-iter(i)}$')
	plt.grid(True)
	plt.legend()
	plt.subplot(122)
	plt.plot(X[:,50], u[:,50],  '--or', label = '$\mathbf{Un_0-iter-max}$' )
	plt.plot(X_1[:,50], u_1[:,50],  '--om', label = '$\mathbf{Un_1-iter-max}$')
	plt.grid(True)
	plt.legend()
	plt.savefig('solut_evol.png')
	plt.show()
	# set up a figure twice as wide as it is tall
	fig = plt.figure(figsize=plt.figaspect(0.5))
	#===============
	# First subplot
	# set up the axes for the first plot
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	# plot a 3D surface like in the example mplot3d/surface3d_demo
	surf0 = ax.plot_surface(X[:,:], Y[:,:], u[:,:], rstride=1, cstride=1, cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
	surf0 = ax.plot_surface(X_1[:,:], Y_1[:,:], u_1[:,:], rstride=1, cstride=1, cmap='viridis',
		               linewidth=0, antialiased=False)
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate solution in uniform mesh')
	ax.set_xlabel('X',  fontweight ='bold')
	ax.set_ylabel('Y',  fontweight ='bold')
	# Add a color bar which maps values to colors.
	fig.colorbar(surf0, shrink=0.5, aspect=25)

	#===============
	# Second subplot
	ax = fig.add_subplot(1, 2, 2, projection='3d')
	surf = ax.plot_surface(X_1[:,:], Y_1[:,:], solut(X_1[:,:], Y_1[:,:]), cmap='viridis',
		               linewidth=0, antialiased=False)
	surf = ax.plot_surface(X[:,:], Y[:,:], solut(X[:,:], Y[:,:]), cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate Solution in adaptive meshes')
	ax.set_xlabel('F1',  fontweight ='bold')
	ax.set_ylabel('F2',  fontweight ='bold')
	fig.colorbar(surf, shrink=0.5, aspect=25)
	plt.savefig('Helmholtz.png')
	plt.show()
