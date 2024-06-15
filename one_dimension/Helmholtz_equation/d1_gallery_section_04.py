__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01']
from pyccel.decorators import types



# assembles mass matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v
    # ...
    
#... utilities of Helmholtz equation
#==============================================================================
# .. in uniform mesh Matrix
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'float', 'double[:,:]')
def assemble_matrix_un_ex01(ne, degree, spans, basis, weights, points, Kappa, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]

                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                                                        
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol - Kappa**2 * bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'float', 'double[:]')
def assemble_vector_ex01(ne, degree, spans, basis, weights, points, vector_d,  Kappa, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import tan
    # ... sizes
    k1 = weights.shape[1]
    # ...
    lcoeffs_d   = zeros(degree+1)
    # ...
    lvalues_u   = zeros(k1)
    lvalues_udx = zeros(k1)
    # ... build rhs
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]

            lcoeffs_d[ : ] = vector_d[i_span_1 : i_span_1+degree+1]
            for g1 in range(0, k1):
                    u  = 0.0
                    ux = 0.0
                    for il_1 in range(0, degree+1):

                            bj_0 = basis[ie1,il_1,0,g1]
                            bj_x = basis[ie1,il_1,1,g1]

                            coeff_d = lcoeffs_d[il_1]

                            u  +=  coeff_d*bj_0
                            ux +=  coeff_d*bj_x
                    lvalues_u[g1]  = u
                    lvalues_udx[g1] = ux
            for il_1 in range(0, degree+1):
                    i1 = i_span_1 - degree + il_1

                    v = 0.0
                    for g1 in range(0, k1):
                            bi_0 = basis[ie1, il_1, 0, g1]
                            bi_x = basis[ie1, il_1, 1, g1]
                            
                            wvol  = weights[ie1, g1]

                            u  = lvalues_u[g1]
                            ux = lvalues_udx[g1]
                            # ...
                            v += Kappa**2*bi_0 * u * wvol - ux * bi_x * wvol

                    rhs[i1+degree] += v   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Robin Condition
    theta = pi/4.
    #.. 
    x1       = 0.
    alpha    = Kappa*x1*cos(theta)
    #rhs[0+degree]       += (- Kappa*cos(theta)*cos(alpha) - Kappa*cos(alpha) )
    #.. 
    x1       = 1.
    alpha    = Kappa*x1*cos(theta)
    #rhs[ne-1+2*degree] += (+ Kappa*cos(theta)*cos(alpha) - Kappa*cos(alpha) )
    # ...


#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'float', 'double[:]')
def assemble_vector_ex11(ne, degree, spans, basis, weights, points, vector_d, Kappa, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import tan
    # ... sizes
    k1          = weights.shape[1]
    # ...
    lcoeffs_d   = zeros(degree+1)
    # ...
    lvalues_u   = zeros(k1)
    lvalues_udx = zeros(k1)
    # ... build rhs
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]

            lcoeffs_d[ : ] = vector_d[i_span_1 : i_span_1+degree+1]
            for g1 in range(0, k1):
                    u  = 0.0
                    ux = 0.0
                    for il_1 in range(0, degree+1):

                            bj_0 = basis[ie1,il_1,0,g1]
                            bj_x = basis[ie1,il_1,1,g1]

                            coeff_d = lcoeffs_d[il_1]

                            u  +=  coeff_d*bj_0
                            ux +=  coeff_d*bj_x
                    lvalues_udx[g1] = ux
                    lvalues_u[g1]   = u
            for il_1 in range(0, degree+1):
                    i1 = i_span_1 - degree + il_1

                    v = 0.0
                    for g1 in range(0, k1):
                            bi_0 = basis[ie1, il_1, 0, g1]
                            bi_x = basis[ie1, il_1, 1, g1]
                            
                            wvol  = weights[ie1, g1]

                            u  = lvalues_u[g1]
                            ux = lvalues_udx[g1]
                            # ...
                            v += Kappa**2 * bi_0 * u * wvol - ux * bi_x * wvol

                    rhs[i1+degree] += v   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Robin Condition
    theta = pi/4.
    #.. 
    x1       = 0.
    alpha    = Kappa*x1*cos(theta)
    #rhs[0+degree] += (Kappa*cos(theta)*sin(alpha) + Kappa*sin(alpha) )
    #.. 
    x1       = 1.
    alpha    = Kappa*x1*cos(theta)
    #rhs[ne-1+2*degree] += (-Kappa*cos(theta)*sin(alpha) + Kappa*sin(alpha) )
    # ...

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'int', 'float', 'double[:]')
def assemble_norm_ex01(ne, degree, spans, basis, weights, points, vector_u, comp, Kappa, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import tan
    # ... sizes
    k1 = weights.shape[1]
    # ...

    lcoeffs_u = zeros(degree+1)
    lvalues_u = zeros(k1)
    lvalues_ux = zeros(k1)

    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]

            lvalues_u[ : ]  = 0.0
            lvalues_ux[ : ] = 0.0
            lcoeffs_u[ : ]  = vector_u[i_span_1 : i_span_1+degree+1]
            for il_1 in range(0, degree+1):
                    coeff_u = lcoeffs_u[il_1]

                    for g1 in range(0, k1):
                        b1  = basis[ie1,il_1,0,g1]
                        db1 = basis[ie1,il_1,1,g1]

                        lvalues_u[g1]  += coeff_u*b1
                        lvalues_ux[g1] += coeff_u*db1

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                    wvol  = weights[ie1, g1]
                    x     = points[ie1, g1]

                    theta = pi/4.
                    alpha    = Kappa*x#*cos(theta)
                    # ... test 0
                    if comp ==0:
                       u   = cos(alpha)
                       ux  = -1.*Kappa*sin(alpha)
                    else:
                       u   = sin(alpha)
                       ux  = Kappa*cos(alpha)
                    #..                    
                    uh  = lvalues_u[g1]
                    uhx = lvalues_ux[g1]

                    v  += (u-uh)**2 * wvol
                    w  += (ux-uhx)**2 * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2   = sqrt(norm_l2)
    norm_H1   = sqrt(norm_H1)

    rhs[degree]   = norm_l2
    rhs[degree+1] = norm_H1
    # ...

