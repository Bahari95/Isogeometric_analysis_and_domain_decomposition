__all__ = ['assemble_stiffnessmatrix1D',
           'assemble_vector_ex01',
           'assemble_norm_ex01'
]

from pyccel.decorators import types

# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
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

#============================================================================== Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'double[:]')
def assemble_vector_ex01(ne1, p1, spans_1,  basis_1,  weights_1, points_1, vector_d, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    # ...
    lcoeffs_d   = zeros(p1+1)
    # ..
    lvalues_u   = zeros(k1)
    lvalues_udx = zeros(k1)

    # ... build rhs
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_d[ : ] = vector_d[i_span_1 : i_span_1+p1+1]
            for g1 in range(0, k1):
                    ux = 0.0
                    for il_1 in range(0, p1+1):
	
                            bj_x    = basis_1[ie1,il_1,1,g1]

                            coeff_d = lcoeffs_d[il_1]

                            ux +=  coeff_d*bj_x
                    lvalues_udx[g1] = ux
                    # ...
                    x    = points_1[ie1, g1]
                    # ...
                    f = pi**2*sin(pi*x)
                    # ..
                    lvalues_u[g1] = f
            for il_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1

                    v = 0.0
                    for g1 in range(0, k1):
                            bi_0  = basis_1[ie1, il_1, 0, g1]
                            bi_x  = basis_1[ie1, il_1, 1, g1]
                            # ...
                            wvol  = weights_1[ie1, g1]
                            # ...
                            u     = lvalues_u[g1]
                            ux    = lvalues_udx[g1]
                            # ... 
                            v    += bi_0 * u * wvol - ux * bi_x * wvol

                    rhs[i1+p1] += v   
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'double[:]', 'real', 'double[:]')
def assemble_vector_ex02(ne1, p1, spans_1,  basis_1,  weights_1, points_1,  knots_1, vector_d, ovlp_value, rhs):
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1        = weights_1.shape[1]
    # ...
    # ...
    lcoeffs_d   = zeros(p1+1)
    #---Computes All basis in a new points
    nders          = 0
    degree         = p1
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis1         = zeros(degree+1)

    #span = find_span( knots, degree, xq )
    xq   = ovlp_value
    #~~~~~~~~~~~~~~~
    # Knot index at left/right boundary
    low  = degree
    high = len(knots_1)-1-degree
    # Check if point is exactly on left/right boundary, or outside domain
    if xq <= knots_1[low ]: 
         span = low
    if xq >= knots_1[high]: 
         span = high-1
    else :
      # Perform binary search
      span = (low+high)//2
      while xq < knots_1[span] or xq >= knots_1[span+1]:
         if xq < knots_1[span]:
             high = span
         else:
             low  = span
         span = (low+high)//2
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = xq - knots_1[span-j]
        right[j] = knots_1[span+1+j] - xq
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
    basis1[:] = ders[0,:]
    # ... build rhs
    #... We compute firstly the span in new adapted points                             
    # ...
    lcoeffs_d[ : ] = vector_d[span : span+p1+1]
    # ...
    u  = 0.
    for il_1 in range(0, p1+1):
        bi_0    = basis1[il_1]
        # ...
        coeff_d  = lcoeffs_d[il_1]
        # ...
        u       +=  coeff_d * bi_0
    rhs[p1]     = u
    # ...    
    

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'double[:]')
def assemble_norm_ex01(ne1, p1, spans_1,  basis_1,  weights_1, points_1, vector_u, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    # ...

    lcoeffs_u  = zeros(p1+1)
    lvalues_u  = zeros(k1 )
    lvalues_ux = zeros(k1 )

    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lvalues_u  [ : ] = 0.0
            lvalues_ux [ : ] = 0.0
            lcoeffs_u[ : ]   = vector_u[i_span_1 : i_span_1+p1+1]
            for il_1 in range(0, p1+1):
                    coeff_u = lcoeffs_u[il_1]
                    for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]

                        lvalues_u [g1] += coeff_u*b1
                        lvalues_ux[g1] += coeff_u*db1

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                    wvol = weights_1[ie1, g1]
                    x    = points_1[ie1, g1]

                    # ... test 0
                    u   = sin(pi*x)
                    ux  = pi*cos(pi*x)
                    # ... test 1
                    uh  = lvalues_u [g1]
                    uhx = lvalues_ux[g1]

                    v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1]   = norm_l2
    rhs[p1+1] = norm_H1
    # ...
