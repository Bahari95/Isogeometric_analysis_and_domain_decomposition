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
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        K, matrix):

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]


                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                                    # ...
                                    v += (bj_x1 * bi_x1 + bj_x2 * bi_x2) * wvol - K**2 * bi_0 * bj_0 * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...
# .. in uniform mesh Matrix
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'double[:,:,:,:]')
def assemble_matrix_un_ex11(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        K, matrix):

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    ie1  = 0
    i_span_1 = spans_1[ie1]
    for ie2 in range(0, ne2):
                i_span_2 = spans_2[ie2]
                    
                for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g2 in range(0, k2):
                                    bi_0  = basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_2[ie2, jl_2, 0, g2]

                                    wvol  = weights_2[ie2, g2]
                                    # ...
                                    v    += K * bi_0 * bj_0 * wvol

                            matrix[p1, p2+i2, p1, p2+j2-i2]  += v
    # ... build matrices
    ie1      = ne1-1
    i_span_1 = spans_1[ie1]
    for ie2 in range(0, ne2):
                i_span_2 = spans_2[ie2]
                    
                for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g2 in range(0, k2):
                                    bi_0  = basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_2[ie2, jl_2, 0, g2]

                                    wvol  = weights_2[ie2, g2]
                                    # ...
                                    v    += K * bi_0 * bj_0 * wvol

                            matrix[i_span_1+p1, p2+i2, p1, p2+j2-i2]  += v
    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]
            ie2      = 0
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                    for jl_1 in range(0, p1+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            v  = 0.0
                            for g1 in range(0, k1):
                                    bi_0  = basis_1[ie1, il_1, 0, g1]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1]

                                    wvol  = weights_1[ie1, g1]
                                    # ...
                                    v    += K* bi_0 * bj_0 * wvol

                            matrix[p1+i1, p2, p1+j1-i1, p2]  += v
    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]
            ie2      = ne2-1
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                    for jl_1 in range(0, p1+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            v  = 0.0
                            for g1 in range(0, k1):
                                    bi_0  = basis_1[ie1, il_1, 0, g1]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1]

                                    wvol  = weights_1[ie1, g1]
                                    # ...
                                    v    += K* bi_0 * bj_0 * wvol

                            matrix[p1+i1, i_span_2+p2, p1+j1-i1, p2]  += v
    # ...
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]',  'float', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_d, K,  rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import tan
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_d   = zeros((p1+1,p2+1))
    # ...
    lvalues_u   = zeros((k1, k2))
    lvalues_udx = zeros((k1, k2))
    lvalues_udy = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_d[ : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    u  = 0.0
                    ux = 0.0
                    uy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_0 = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                            bj_x = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                            coeff_d = lcoeffs_d[il_1,il_2]

                            u  +=  coeff_d*bj_0
                            ux +=  coeff_d*bj_x
                            uy +=  coeff_d*bj_y
                    lvalues_udx[g1, g2] = ux
                    lvalues_udy[g1, g2] = uy
                    lvalues_u  [g1,g2]  = u
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u  = lvalues_u[g1,g2]
                            ux = lvalues_udx[g1,g2]
                            uy = lvalues_udy[g1,g2]
                            # ...
                            v += K**2*bi_0 * u * wvol - (ux * bi_x +uy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Robin Condition
    theta = pi/4.
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]        
        for il_1 in range(0, p1+1):
           i1    = i_span_1 - p1 + il_1

           vx_0 = 0.0
           vx_1 = 0.0
           for g1 in range(0, k1):
                  bi_0     =  basis_1[ie1, il_1, 0, g1]
                  wleng_x  =  weights_1[ie1, g1]
                  x1       =  points_1[ie1, g1]
                  #..
                  x2       = 0.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vx_0    += bi_0*( - K*sin(theta)*cos(alpha) - K*cos(alpha) )* wleng_x
                  #..
                  x2       = 1.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vx_1    += bi_0*(+ K*sin(theta)*cos(alpha) - K*cos(alpha) ) * wleng_x

           rhs[i1+p1,0+p2]       += vx_0
           rhs[i1+p1,ne2+2*p2-1] += vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0     =  basis_2[ie2, il_2, 0, g2]
                  wleng_y  =  weights_2[ie2, g2]
                  x2       =  points_2[ie2, g2]
                  #.. 
                  x1       = 0.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vy_0    += bi_0*(- K*cos(theta)*cos(alpha) - K*cos(alpha) ) * wleng_y
                  #.. 
                  x1       = 1.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vy_1    += bi_0*(+ K*cos(theta)*cos(alpha) - K*cos(alpha) ) * wleng_y

           rhs[0+p1,i2+p2]       += vy_0
           rhs[ne1-1+2*p1,i2+p2] += vy_1
    # ...


#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'float', 'double[:,:]')
def assemble_vector_ex11(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_d, K,  rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import tan
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_d   = zeros((p1+1,p2+1))
    # ...
    lvalues_u   = zeros((k1, k2))
    lvalues_udx = zeros((k1, k2))
    lvalues_udy = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_d[ : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    u  = 0.0
                    ux = 0.0
                    uy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_0 = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                            bj_x = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                            coeff_d = lcoeffs_d[il_1,il_2]

                            u  +=  coeff_d*bj_0
                            ux +=  coeff_d*bj_x
                            uy +=  coeff_d*bj_y
                    lvalues_udx[g1, g2] = ux
                    lvalues_udy[g1, g2] = uy
                    lvalues_u  [g1,g2]  = u
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u  = lvalues_u[g1,g2]
                            ux = lvalues_udx[g1,g2]
                            uy = lvalues_udy[g1,g2]
                            # ...
                            v += K**2*bi_0 * u * wvol - (ux * bi_x + uy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Robin Condition
    theta = pi/4.
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]        
        for il_1 in range(0, p1+1):
           i1    = i_span_1 - p1 + il_1

           vx_0 = 0.0
           vx_1 = 0.0
           for g1 in range(0, k1):
                  bi_0     =  basis_1[ie1, il_1, 0, g1]
                  wleng_x  =  weights_1[ie1, g1]
                  x1       =  points_1[ie1, g1]
                  #..
                  x2       = 0.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vx_0    += bi_0*(K*sin(theta)*sin(alpha) + K*sin(alpha) )* wleng_x
                  #..
                  x2       = 1.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vx_1    += bi_0*(-K*sin(theta)*sin(alpha) + K*sin(alpha) ) * wleng_x

           rhs[i1+p1,0+p2]       += vx_0
           rhs[i1+p1,ne2+2*p2-1] += vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0     =  basis_2[ie2, il_2, 0, g2]
                  wleng_y  =  weights_2[ie2, g2]
                  x2       =  points_2[ie2, g2]
                  #.. 
                  x1       = 0.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vy_0    += bi_0*(K*cos(theta)*sin(alpha) + K*sin(alpha) ) * wleng_y
                  #.. 
                  x1       = 1.
                  alpha    = K*(x1*cos(theta) + x2*sin(theta))
                  vy_1    += bi_0*(-K*cos(theta)*sin(alpha) + K*sin(alpha) ) * wleng_y

           rhs[0+p1,i2+p2]       += vy_0
           rhs[ne1-1+2*p1,i2+p2] += vy_1
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'real', 'double[:]')
def assemble_vector_ex02(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2,  spans_3, basis_1, basis_2, basis_3, weights_1, weights_2, weights_3, points_1, points_2, points_3,  knots_1, knots_2, knots_3, vector_d, ovlp_value, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1        = weights_1.shape[1]
    k2        = weights_2.shape[1]
    # ...
    # ...
    lcoeffs_d   = zeros((p2+1,p3+1))
    # ..
    lvalues_u   = zeros(k1)
    #---Computes All basis in a new points
    nders          = 0
    degree         = p2
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis2         = zeros(degree+1)
    for i in range(1):
            #span = find_span( knots, degree, xq )
            xq = ovlp_value
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_2)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_2[low ]: 
                 span = low
            elif xq >= knots_2[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_2[span] or xq >= knots_2[span+1]:
                 if xq < knots_2[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_2[span-j]
                right[j] = knots_2[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis2[:] = ders[0,:]
    # ... build rhs
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            for g1 in range(0, k1):

                    #... We compute firstly the span in new adapted points                             
                    #...                    
                    xq        = ovlp_value
                    degree    = p2
                    low       = degree
                    high      = len(knots_2)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_2[low ]: 
                         span = low
                    elif xq >= knots_2[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_2[span] or xq >= knots_2[span+1]:
                         if xq < knots_2[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_2    = span 
                    # ...
                    lcoeffs_d[:, : ] = vector_d[span_2 : span_2+p2+1, i_span_1 : i_span_1+p3+1]
                    # ...
                    u  = 0.
                    for il_1 in range(0, p2+1):
                       for il_2 in range(0, p3+1):
                             bi_0    = basis2[il_1] * basis_3[ie1, il_2, 0, g1]
                             # ...
                             coeff_d = lcoeffs_d[il_1, il_2]
                             # ...
                             u      +=  coeff_d*bi_0
                    lvalues_u[g1] = u

            for il_1 in range(0, p1+1):

                      i1 = i_span_1 - p1 + il_1

                      v = 0.0
                      for g1 in range(0, k1):

                              bi_0 = basis_1[ie1, il_1, 0, g1]
                              # ...
                              wvol = weights_1[ie1, g1]
                              # ...
                              u    = lvalues_u[g1]
                              # ...        
                              v   += bi_0 * u * wvol

                      rhs[i1+p1] += v   
    # ...    

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'int', 'float', 'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, comp, K, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import tan
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    lcoeffs_u = zeros((p1+1,p2+1))
    lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ]  = 0.0
            lvalues_ux[ : , : ] = 0.0
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]  += coeff_u*b1*b2
                            lvalues_ux[g1,g2] += coeff_u*db1*b2
                            lvalues_uy[g1,g2] += coeff_u*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    x     = points_1[ie1, g1]
                    y     = points_2[ie2, g2]
                    theta = pi/4.
                    alpha    = K*(x*cos(theta) + y*sin(theta))
                    # ... test 0
                    if comp ==0:
                       u   = cos(alpha)
                       ux  = -1.*K*cos(theta)*sin(alpha)
                       uy  = -1.*K*sin(theta)*sin(alpha)
                    else:
                       u   = sin(alpha)
                       ux  = K*cos(theta)*cos(alpha)
                       uy  = K*sin(theta)*cos(alpha)
                    #..                    
                    uh  = lvalues_u[g1,g2]
                    uhx = lvalues_ux[g1,g2]
                    uhy = lvalues_uy[g1,g2]

                    v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2+(uy-uhy)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1,p2] = norm_l2
    rhs[p1,p2+1] = norm_H1
    # ...

