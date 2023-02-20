__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_vector_ex02',
           'assemble_norm_ex01'
]

#==============================================================================
from pyccel.decorators import types

# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

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
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v

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


#==============================================================================
#
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]','double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w, vector_v, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    rho = 3.

    # ...
    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p1+1,p2+1))
    lcoeffs_v = zeros((p1+1,p2+1))
    lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))


    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            #... Integration of Dirichlet boundary conditions
            lcoeffs_v[ : , : ] = vector_v[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            # ...
                            coeff_u   = lcoeffs_v[il_1,il_2]
                            # ...
                            sx      +=  coeff_u*bj_x
                            sy      +=  coeff_u*bj_y
                    lvalues_ux[g1,g2] = sx
                    lvalues_uy[g1,g2] = sy

            lvalues_u[ : , : ] = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]            
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                    ty = 0.0
                    x  = 0.0
                    y  = 0.0
                    #/:
                    sx = 0.0
                    sy = 0.0
                    #..
                    sxx = 0.0
                    syy = 0.0
                    sxy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):
                              bj_0 = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              # ...
                              bj_x = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              # ...
                              bj_xx = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              bj_xy = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]
                              #...
                              coeff_u = lcoeffs_u[il_1,il_2]
                              sx     +=  coeff_u*bj_x
                              sy     +=  coeff_u*bj_y
                              sxx    +=  coeff_u*bj_xx
                              syy    +=  coeff_u*bj_yy
                              sxy    +=  coeff_u*bj_xy
                              #...
                              coeff_w = lcoeffs_w[il_1,il_2]
                              ty +=  coeff_w*bj_y
                              x  +=  coeff_u*bj_0
                              y  +=  coeff_w*bj_0                                                            
                    #.. 
                    G_u = sqrt(sx**2 + ty**2 + 2.*sy**2 + 2.*rho)
                    DG_u= sxx*sx + syy*ty + 2.*sxy*sy 
                    lvalues_u[g1,g2] = -DG_u/G_u 
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x  = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            #...
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ...Dirichlet boundary conditions
                            ux   = lvalues_ux[g1,g2]
                            uy   = lvalues_uy[g1,g2]
                            #..
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol - (ux * bi_x + uy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...

@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]')
def assemble_vector_ex02(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w,  vector_v, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    rho = 3.
    #...
    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p1+1,p2+1))
    lcoeffs_v = zeros((p1+1,p2+1))
    lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
            
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            #... Integration of Dirichlet boundary conditions
            lcoeffs_v[ : , : ] = vector_v[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            # ...
                            coeff_u   = lcoeffs_v[il_1,il_2]
                            # ...
                            sx      +=  coeff_u*bj_x
                            sy      +=  coeff_u*bj_y
                    lvalues_ux[g1,g2] = sx
                    lvalues_uy[g1,g2] = sy
                    
            lvalues_u[ : , : ] = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]            
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                    sx = 0.0
                    x  = 0.
                    y  = 0.
                    #/:
                    tx = 0.0
                    ty = 0.0
                    #..
                    txx = 0.0
                    tyy = 0.0
                    txy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0 = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              # ...
                              bj_x = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              # ...
                              bj_xx = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              bj_xy = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]
                              #...
                              coeff_w = lcoeffs_w[il_1,il_2]
                              tx     +=  coeff_w*bj_x
                              ty     +=  coeff_w*bj_y
                              txx    +=  coeff_w*bj_xx
                              tyy    +=  coeff_w*bj_yy
                              txy    +=  coeff_w*bj_xy
                              #...
                              coeff_u = lcoeffs_u[il_1,il_2]
                              sx +=  coeff_u*bj_x
                              x  +=  coeff_u*bj_0
                              y  +=  coeff_w*bj_0
                                                            
                    #.. 
                    G_u = sqrt(sx**2 + ty**2 + 2.*tx**2 + 2.*rho)
                    DG_u= txx*sx + tyy*ty + 2.*txy*tx
                    lvalues_u[g1,g2] = -DG_u/G_u
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x  = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            #...
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ...Dirichlet boundary conditions
                            ux   = lvalues_ux[g1,g2]
                            uy   = lvalues_uy[g1,g2]
                            #..
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol - (ux * bi_x + uy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...
    
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'real', 'double[:]')
def assemble_vector_ex12(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2,  spans_3, basis_1, basis_2, basis_3, weights_1, weights_2, weights_3, points_1, points_2, points_3,  knots_1, knots_2, knots_3, vector_d, ovlp_value, rhs):

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
            if xq >= knots_2[high]: 
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
                    if xq >= knots_2[high]: 
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
#==============================================================================
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]','double[:,:]')
def assemble_residual_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, rhs):

    from numpy import exp
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    lcoeffs_u  = zeros((p1+1,p2+1))
    lvalues_u  = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    Res_H1 = 0.
    Res_l2 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ] = 0.0
            lvalues_ux[ : , : ] = 0.0
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2 = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]  += coeff_u*b1*b2
                            lvalues_ux[g1,g2] += coeff_u*db1*b2
                            lvalues_uy[g1,g2] += coeff_u*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    uh  = lvalues_u[g1,g2]
                    uhx = lvalues_ux[g1,g2]
                    uhy = lvalues_uy[g1,g2]

                    v  += (uhx**2+uhy**2) * wvol
                    w  += uh**2 * wvol

            Res_H1 += v
            Res_l2 += w
    Res_H1 = sqrt(Res_H1)
    Res_l2 = sqrt(Res_l2)
    rhs[p1,p2] = Res_H1
    rhs[p1,p2+1] = Res_l2
    # ...
#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p2+1,p1+1))
    #lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    #norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_2[ie1]
        for ie2 in range(0, ne2):
            i_span_3 = spans_2[ie2]
            i_span_2 = spans_1[ie2]

            lvalues_ux[ : , : ] = 0.0
            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] #M^p2-1

                            lvalues_ux[g1,g2] += coeff_u*b1*b2
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_w[ : , : ]  = vector_w[i_span_4 : i_span_4+p2+1, i_span_2 : i_span_2+p1+1]
            for il_1 in range(0, p2+1):
                for il_2 in range(0, p1+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_2[ie1,il_1,0,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_1[ie2,il_2,0,g2] 

                            lvalues_uy[g1,g2] += coeff_w*b1*b2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]

                    # ... test 1
                    #u  =  sin(pi*x1)*sin(pi*x2)
                    #ux =  ( 0.2+0.8*x)*cos(0.5*pi*y)
                    #uy =  ( 0.2+0.8*x)*sin(0.5*pi*y)
                    # ... test 2
                    #u  =   exp((1-2.*x)**2)*cos(pi*y**2)
                    ux =  0.15*sin(2.*pi*(y+0.1))*x+x
                    uy =  0.15*sin(2.*pi*(x+0.1))*y+y
                    # ... test 3
                    #u  =   x1*exp((x1**2+x2**2)/2.)
                    #ux =   0.5*(2.*x-1.)*sqrt(1.-(2.*y-1.)**2/2.0)
                    #uy =   0.5*(2.*y-1.)*sqrt(1.-(2.*x-1.)**2/2.0)

                    #uh  = lvalues_u[g1,g2]
                    uhx = lvalues_ux[g1,g2]
                    uhy = lvalues_uy[g1,g2]

                    #v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2 + (uy-uhy)**2) * wvol
            #norm_l2 += v
            norm_H1 += w

    #norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    #rhs[p1,p2] = norm_l2
    rhs[p1,p2+1] = norm_H1
    # ...
#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_det_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p2+1,p1+1))
    #lvalues_u = zeros((k1, k2))
    lvalues_vx = zeros((k1, k2))
    lvalues_vy = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
    
    min_det = 5.
    max_det = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_2[ie1]
        for ie2 in range(0, ne2):
            i_span_3 = spans_2[ie2]
            i_span_2 = spans_1[ie2]

            lvalues_ux[ : , : ] = 0.0
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] #M^p2-1
                            db2  = basis_2[ie2,il_2,1,g2] #M^p2-1

                            lvalues_ux[g1,g2] += coeff_u*db1*b2
                            lvalues_uy[g1,g2] += coeff_u*b1*db2
            lvalues_vx[ : , : ] = 0.0
            lvalues_vy[ : , : ] = 0.0
            lcoeffs_w[ : , : ]  = vector_w[i_span_4 : i_span_4+p2+1, i_span_2 : i_span_2+p1+1]
            for il_1 in range(0, p2+1):
                for il_2 in range(0, p1+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_2[ie1,il_1,0,g1] #M^p1-1
                        db1  = basis_2[ie1,il_1,1,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_1[ie2,il_2,0,g2] 
                            db2  = basis_1[ie2,il_2,1,g2] 
                            
                            lvalues_vx[g1,g2] += coeff_w*db1*b2
                            lvalues_vy[g1,g2] += coeff_w*b1*db2

            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    uhx = lvalues_ux[g1,g2]
                    uhy = lvalues_uy[g1,g2]
                    #...
                    vhx = lvalues_vx[g1,g2]
                    vhy = lvalues_vy[g1,g2]

                    det_Jac_map = uhx*vhy-uhy*vhx
                    if max_det < det_Jac_map :
                        max_det = det_Jac_map
                    if min_det > det_Jac_map :
                        min_det = det_Jac_map

    rhs[p1,p2]   = min_det
    rhs[p1,p2+1] = max_det
    # ... 
