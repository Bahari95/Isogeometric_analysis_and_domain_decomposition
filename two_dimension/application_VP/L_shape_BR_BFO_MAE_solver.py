from simplines import compile_kernel

#from spaces import SplineSpace
#from spaces import TensorSpace

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import least_square_Bspline

#from matplotlib.pyplot import plot, show

import matplotlib.pyplot as plt
#%matplotlib inline

from gallery_section_01 import assemble_vector_ex01
from gallery_section_01 import assemble_vector_ex02
from gallery_section_01 import assemble_residual_ex01
from gallery_section_01 import assemble_vector_ex12    #---1 : In uniform mesh

assemble_Pr         = compile_kernel(assemble_vector_ex12, arity=1)
assemble_rhs01      = compile_kernel(assemble_vector_ex01, arity=1)
assemble_rhs10      = compile_kernel(assemble_vector_ex02, arity=1)
assemble_residual   = compile_kernel(assemble_residual_ex01, arity=1)

from gallery_section_01 import assemble_stiffnessmatrix1D
from gallery_section_01 import assemble_massmatrix1D
assemble_stiffness = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass = compile_kernel( assemble_massmatrix1D, arity=2)


from gallery_section_01 import assemble_norm_ex01 #---1 : In uniform mesh
assemble_norm_l2 = compile_kernel(assemble_norm_ex01, arity=1)
#..
from gallery_section_01 import assemble_det_ex01
Test_det = compile_kernel(assemble_det_ex01, arity=1)

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
from   kronecker.fast_diag          import Poisson
import time


#==============================================================================       
def   Pr_h_solve(V1, V2, V, Vt, u, domain_nb, ovlp_value): 

       # Stiffness and Mass matrix in 1D in the first deriction
       M1         = assemble_mass(V1)
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
       
       return x_n
       
#.......Picard BFO ALGORITHM
def picard(V1, V2, V, u01, u10, u_01= None, u_10= None, x0 = None, y0 = None):

    niter = 50
    tol = 1e-14
    from numpy import zeros
 
    du = StencilVector(V.vector_space)
    if u_10 is None :
       x1 = zeros(V.nbasis)
       u_01 = StencilVector(V.vector_space)   
       u_10 = StencilVector(V.vector_space)
    if u_10 is not None : 
       x1 = u_10.toarray().reshape(V.nbasis)
    #... We delete the first and the last spline function
    #. as a technic for applying Dirichlet boundary condition

    #..Stiffness and Mass matrix in 1D in the first deriction
    K1 = assemble_stiffness(V1)
    K1 = K1.tosparse()
    K1 = K1.toarray()[1:-1,1:-1]
    K1 = csr_matrix(K1)

    M1 = assemble_mass(V1)
    M1 = M1.tosparse()
    M1 = M1.toarray()[1:-1,1:-1]
    M1 = csr_matrix(M1)

    # Stiffness and Mass matrix in 1D in the second deriction
    K2 = assemble_stiffness(V2)
    K2 = K2.tosparse()
    K2 = K2.toarray()[1:-1,1:-1]
    K2 = csr_matrix(K2)

    M2 = assemble_mass(V2)
    M2 = M2.tosparse()
    M2 = M2.toarray()[1:-1,1:-1]
    M2 = csr_matrix(M2)

    mats_1 = [M1, K1]
    mats_2 = [M2, K2]

    # ...
    poisson = Poisson(mats_1, mats_2)

    for i in range(niter):
        #--Assembles a right hand side of Poisson equation
        rhs = assemble_rhs01( V , fields=[u_01, u_10, u01])
        b   = rhs.toarray()
        
        b   = b.reshape(V.nbasis) 
        b   = b[1:-1, 1:-1]      
        b   = b.reshape((V1.nbasis-2)*(V1.nbasis-2))

        #...
        xkron  = poisson.solve(b)
        #...
        xkron  = xkron.reshape([V1.nbasis-2,V2.nbasis-2])
        x      = zeros(V.nbasis)
        x[1:-1, 1:-1] = xkron[:, :]
        
        # ...Dirichlet
        x[:, :] += x0[:, :]
        x01      = x           
        
        #//
        #--Assembles a right hand side of Poisson equation
        rhs = assemble_rhs10( V , fields=[u_01, u_10, u10])
        b   = rhs.toarray()
        b   = b.reshape(V.nbasis) 
        b   = b[1:-1, 1:-1]      
        b   = b.reshape((V1.nbasis-2)*(V2.nbasis-2))
        
        #...
        xkron  = poisson.solve(b)
        #...
        xkron  = xkron.reshape([V1.nbasis-2,V2.nbasis-2])
        x      = zeros(V.nbasis)
        x[1:-1, 1:-1] = xkron
        # ...Dirichlet
        x[:, :] += y0[:, :]
        x10 = x
        
        #... update the unkowns
        u_01.from_array(V, x01) 
        u_10.from_array(V, x10)         
        #//        
        dx = x-x1
        x1 = x
        
        du.from_array(V, dx)

        # Compute residual for L2 and H1 norm
        Res = assemble_residual(V, fields=[du])
        resid= Res.toarray()
        H1_residual = resid[0]
        l2_residual = resid[1]

        if l2_residual < tol:
            break
    #print(b)
    return u_01, u_10, x01, x10, i, l2_residual, H1_residual

degree    = 2
nelements = 16
alpha     = 0.75
beta      = 0.25
#..... Initialisation and computing optimal mapping for 16*16
#-------------------------------------------DOmaine_0
# create the spline space for each direction
grids_0 = linspace(0., alpha, nelements+1)
V1_0    = SplineSpace(degree=degree, nelements= nelements, grid = grids_0, nderiv = 2)
V2_0    = SplineSpace(degree=degree, nelements= nelements, nderiv = 2)
# create the tensor space
Vh_0    = TensorSpace(V1_0, V2_0)

#-------------------------------------------DOmaine_0
grids_1 = linspace(beta, 1., nelements+1)
V1_1    = SplineSpace(degree=degree, nelements= nelements, grid =grids_1, nderiv = 2)
V2_1    = SplineSpace(degree=degree, nelements= nelements, nderiv = 2)
# create the tensor space
Vh_1    = TensorSpace(V1_1, V2_1)

#...
Vt_0    = TensorSpace(V2_0, V1_1, V2_1)
Vt_1    = TensorSpace(V2_1, V1_0, V2_0)

#------------------------------
# compute the interpolate spline function for tyhe Dirichlet boundary condition

sol_dx  = lambda x,y : x
sol1_dx = lambda x,y : x
sol_dy  = lambda x,y : 2.*y


#---------------------------------------------------DOmaine_0
fx0 = lambda y : sol_dx(0.,y)
fx1 = lambda y : sol1_dx(alpha,y)
fy0 = lambda x : sol_dx(x,0.)
fy1 = lambda x : sol_dx(x,1.)
#__
gx0 = lambda y : sol_dy(0.,y)
gx1 = lambda y : sol_dy(alpha,y)
gy0 = lambda x : sol_dy(x,0.)
gy1 = lambda x : sol_dy(x,1.)
#__


u01_0   = StencilVector(Vh_0.vector_space)
u10_0   = StencilVector(Vh_0.vector_space)

xD_0    = np.zeros(Vh_0.nbasis)
yD_0    = np.zeros(Vh_0.nbasis)

xD_0[0, : ]              = least_square_Bspline(V2_0.degree, V2_0.knots, fx0)
xD_0[V1_0.nbasis-1, : ]  = least_square_Bspline(V2_0.degree, V2_0.knots, fx1)
xD_0[:,0]                = least_square_Bspline(V1_0.degree, V1_0.knots, fy0)
xD_0[:, V2_0.nbasis - 1] = least_square_Bspline(V1_0.degree, V1_0.knots, fy1)

yD_0[0, : ]              = least_square_Bspline(V2_0.degree, V2_0.knots, gx0)
yD_0[V1_0.nbasis-1, : ]  = least_square_Bspline(V2_0.degree, V2_0.knots, gx1)
yD_0[:,0]                = least_square_Bspline(V1_0.degree, V1_0.knots, gy0)
yD_0[:, V2_0.nbasis - 1] = least_square_Bspline(V1_0.degree, V1_0.knots, gy1)

u01_0.from_array(Vh_0, xD_0)
u10_0.from_array(Vh_0, yD_0)


#---------------------------------------------------DOmaine_1
sol_dx  = lambda x,y : 2.*x
sol1_dx = lambda x,y : 2.*x
sol_dy  = lambda x,y : y

fx0 = lambda y : sol1_dx(beta,y)
fx1 = lambda y : sol_dx(1.,y)
fy0 = lambda x : sol_dx(x,0.)
fy1 = lambda x : sol_dx(x,1.)
#__
gx0 = lambda y : sol_dy(beta,y)
gx1 = lambda y : sol_dy(1.,y)
gy0 = lambda x : sol_dy(x,0.)
gy1 = lambda x : sol_dy(x,1.)
#__

u01_1   = StencilVector(Vh_1.vector_space)
u10_1   = StencilVector(Vh_1.vector_space)

xD_1    = np.zeros(Vh_1.nbasis)
yD_1    = np.zeros(Vh_1.nbasis)

xD_1[0, : ]              = least_square_Bspline(V2_1.degree, V2_1.knots, fx0)
xD_1[V1_1.nbasis-1, : ]  = least_square_Bspline(V2_1.degree, V2_1.knots, fx1)
xD_1[:,0]                = least_square_Bspline(V1_1.degree, V1_1.knots, fy0)
xD_1[:, V2_1.nbasis - 1] = least_square_Bspline(V1_1.degree, V1_1.knots, fy1)

yD_1[0, : ]              = least_square_Bspline(V2_1.degree, V2_1.knots, gx0)
yD_1[V1_1.nbasis-1, : ]  = least_square_Bspline(V2_1.degree, V2_1.knots, gx1)
yD_1[:,0]                = least_square_Bspline(V1_1.degree, V1_1.knots, gy0)
yD_1[:, V2_1.nbasis - 1] = least_square_Bspline(V1_1.degree, V1_1.knots, gy1)

u01_1.from_array(Vh_1, xD_1)
u10_1.from_array(Vh_1, yD_1)

#np.savetxt('L_shape/Cp_TNSx_'+str(degree)+'_'+str(nelements)+'_'+str(numberpatchs)+'.txt', xD, fmt='%.2e')
#np.savetxt('L_shape/Cp_TNSy_'+str(degree)+'_'+str(nelements)+'_'+str(numberpatchs)+'.txt', yD, fmt='%.2e')
#+++++++++++++++++++++++++++++++++++
print('++Mixed-formulation-BFO-PICARD--------------------BAHARI-RATNANI-NEW-FORMULATION--FOR---VOLUMETRIC-PARAMETERIZATION')
start = time.time()
u11_0_pH, u12_0_pH, x11uh_0, x12uh_0, i_0, l2_residual_0, H1_residual_0 = picard(V1_0, V2_0, Vh_0, u01_0, u10_0, x0 = xD_0, y0 = yD_0)
cpu_time =  time.time()-start

# ...
Det     = Test_det(Vh_0, fields=[u11_0_pH, u12_0_pH])
test_det= Det.toarray()
min_val = test_det[0]
max_val = test_det[1]
print('\n number of iteration = ',i_0,'\n minimum value of determinant =', min_val,'\n maximum value of determinant = ',  max_val , 'l2_residual', l2_residual_0,'H1_residual', H1_residual_0)
#++++++++++++
start = time.time()
u11_1_pH, u12_1_pH, x11uh_1, x12uh_1, i_1, l2_residual_1, H1_residual_1 = picard(V1_1, V2_1, Vh_1, u01_1, u10_1, x0 = xD_1, y0 = yD_1)
cpu_time_1 =  time.time()-start

# ...
Det     = Test_det(Vh_0, fields=[u11_1_pH, u12_1_pH])
test_det= Det.toarray()
min_val = test_det[0]
max_val = test_det[1]
print('\n number of iteration = ',i_1,'\n minimum value of determinant =', min_val,'\n maximum value of determinant = ',  max_val , 'l2_residual', l2_residual_1,'H1_residual', H1_residual_1)

iter_max = 0
for i in range(iter_max):
	# ... Dirichlezt boudndary condition in x = 0.75 and 0.25
	xh_d                             = Pr_h_solve(V2_0, V2_1, Vh_0, Vt_0, u11_1_pH, 0, alpha)
	xD_0[V1_0.nbasis-1, : ]          = xh_d[V1_0.nbasis-1, : ]
	u01_0.from_array(Vh_0, xD_0)
	xh_d                             = Pr_h_solve(V2_0, V2_1, Vh_0, Vt_0, u12_1_pH, 0, alpha)
	yD_0[V1_0.nbasis-1, : ]          = xh_d[V1_0.nbasis-1, : ]
	u10_0.from_array(Vh_0, yD_0)
	# **** domain_1
	xh                               = Pr_h_solve(V2_1, V2_0, Vh_1, Vt_1, u11_0_pH, 1, beta)
	xD_1[0, : ]                      = xh[0, : ]
	u01_1.from_array(Vh_1, xD_1)
	xh                               = Pr_h_solve(V2_1, V2_0, Vh_1, Vt_1, u12_0_pH, 1, beta)
	yD_1[0, : ]                      = xh[0, : ]
	u01_1.from_array(Vh_1, xD_1)
	#...
	#+++++++++++++++++++++++++++++++++++
	print('++Mixed-formulation-BFO-PICARD--------------------BAHARI-RATNANI-NEW-FORMULATION--FOR---VOLUMETRIC-PARAMETERIZATION')
	start = time.time()
	u11_0_pH, u12_0_pH, x11uh_0, x12uh_0, i_0, l2_residual_0, H1_residual_0 = picard(V1_0, V2_0, Vh_0, u01_0, u10_0, x0 = xD_0, y0 = yD_0)
	cpu_time =  time.time()-start

	# ...
	Det     = Test_det(Vh_0, fields=[u11_0_pH, u12_0_pH])
	test_det= Det.toarray()
	min_val = test_det[0]
	max_val = test_det[1]
	print('\n number of iteration = ',i_0,'\n minimum value of determinant =', min_val,'\n maximum value of determinant = ',  max_val , 'l2_residual', l2_residual_0,'H1_residual', H1_residual_0)
	#++++++++++++
	start = time.time()
	u11_1_pH, u12_1_pH, x11uh_1, x12uh_1, i_1, l2_residual_1, H1_residual_1 = picard(V1_1, V2_1, Vh_1, u01_1, u10_1, x0 = xD_1, y0 = yD_1)
	cpu_time_1 =  time.time()-start

	# ...
	Det     = Test_det(Vh_0, fields=[u11_1_pH, u12_1_pH])
	test_det= Det.toarray()
	min_val = test_det[0]
	max_val = test_det[1]
	print('\n number of iteration = ',i_1,'\n minimum value of determinant =', min_val,'\n maximum value of determinant = ',  max_val , 'l2_residual', l2_residual_1,'H1_residual', H1_residual_1)

#________________________________________________________________________________________________________________________
print('->  degree = {}  nelement = {}  CPU-time = {}'.format(degree, nelements, cpu_time + cpu_time_1))
#---Compute a solution
nbpts = 100
#---Solution in uniform mesh
#u  = sol_field_2d((nbpts,nbpts),  x2uh , V11.knots, V11.degree)[1]
ux, a, b, X, Y       = pyccel_sol_field_2d((nbpts,nbpts),  x11uh_0 , Vh_0.knots, Vh_0.degree)
uy, c, d             = pyccel_sol_field_2d((nbpts,nbpts),  x12uh_0 , Vh_0.knots, Vh_0.degree)[:-2]

ux_1, a_1, b_1, Y, X = pyccel_sol_field_2d((nbpts,nbpts),  x11uh_1 , Vh_1.knots, Vh_1.degree)
uy_1, c_1, d_1       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh_1 , Vh_1.knots, Vh_1.degree)[:-2]

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
    phidx = ux[:,i]
    phidy = uy[:,i]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(nbpts):
    phidx = ux[i,:]
    phidy = uy[i,:]

    axes[0].plot(phidx, phidy, '-b', linewidth = 0.25)

#axes[0].set_title( 'Physical domain ' )
for i in range(nbpts):
    phidx = ux_1[:,i]
    phidy = uy_1[:,i]

    axes[0].plot(phidx, phidy, '-k', linewidth = 0.25)
for i in range(nbpts):
    phidx = ux_1[i,:]
    phidy = uy_1[i,:]

    axes[0].plot(phidx, phidy, '-k', linewidth = 0.25)
    
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
i=0
phidx = ux[:,i]
phidy = uy[:,i]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
i=nbpts-1
phidx = ux[:,i]
phidy = uy[:,i]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
#''
i=0
phidx = ux[i,:]
phidy = uy[i,:]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
i=nbpts-1
phidx = ux[i,:]
phidy = uy[i,:]
axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
#axes[0].axis('off')
axes[0].margins(0,0)
# ...
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
i=0
phidx = ux_1[:,i]
phidy = uy_1[:,i]
axes[0].plot(phidx, phidy, '-y', linewidth = 2.)
i=nbpts-1
phidx = ux_1[:,i]
phidy = uy_1[:,i]
axes[0].plot(phidx, phidy, '-y', linewidth = 2.)
#''
i=0
phidx = ux_1[i,:]
phidy = uy_1[i,:]
axes[0].plot(phidx, phidy, '-y', linewidth = 2.)
i=nbpts-1
phidx = ux_1[i,:]
phidy = uy_1[i,:]
axes[0].plot(phidx, phidy, '-y', linewidth = 2.)
#axes[0].axis('off')
axes[0].margins(0,0)
# ...
#axes[0].set_title( 'Physical domain ' )

for i in range(nbpts):
    phidx = ux[:,i]
    phidy = uy[:,i]

    axes[1].plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(nbpts):
    phidx = ux[i,:]
    phidy = uy[i,:]

    axes[1].plot(phidx, phidy, '-b', linewidth = 0.25)

for i in range(nbpts):
    phidx = ux_1[:,i]
    phidy = uy_1[:,i]

    axes[1].plot(phidx, phidy, '-k', linewidth = 0.25)
for i in range(nbpts):
    phidx = ux_1[i,:]
    phidy = uy_1[i,:]

    axes[1].plot(phidx, phidy, '-k', linewidth = 0.25)
    
axes[1].plot(u11_1_pH.toarray(), u12_1_pH.toarray(), 'ro')
axes[1].axis('off')
#axes[0].margins(0,0)
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('meshes_examples.png')
plt.show()

figtitle        = 'THE_TWO_component'
fig, axes       = plt.subplots( 1, 2, figsize=[12,12], gridspec_kw={'width_ratios': [2, 2]} , num=figtitle )
for ax in axes:
   ax.set_aspect('equal')

axes[0].set_title( 'first component of of approximate solution' )
im = axes[0].contourf( X, Y, ux, cmap= 'jet')
fig.colorbar(im, ax=axes[0], shrink=0.75, aspect=50)
axes[1].set_title( 'second component of approximate solution' )
ima = axes[1].contourf( X, Y, uy, cmap= 'jet')
fig.colorbar(ima, ax=axes[1], shrink=0.75, aspect=50)
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('mixed_solution.png')
plt.show()

#---------------------------------------------------------
#..3Dsurface
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax.set_xlabel('$\mathrm{x}$')
ax.set_ylabel('$\mathrm{y}$')
ax.set_zlabel('$J(\mathbf{u})$')
plt.savefig('Jacobian_function.png')
plt.show()

