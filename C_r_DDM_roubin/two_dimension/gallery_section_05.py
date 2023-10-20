__all__ = ['assemble_stiffnessmatrix1D',
           'assemble_massmatrix1D',
           'assemble_norm_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex02',
           'assemble_norm_ex02'
]

from pyccel.decorators import types

# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            # evaluation dependant uniquement de l'element
            for il_1 in range(0, degree+1):
                for il_2 in range(0, degree+1):
                            
                            for g1 in range(0, k1):
                               
                                    i_span_1 = spans[ie1, g1]
                                    i1 = i_span_1 - degree + il_1
                                    i2 = i_span_1 - degree + il_2
                                    
                                    
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    matrix[ degree+ i1, degree+ i2-i1]  += bi_x * bj_x * wvol

# assembles mass matrix 1D
#==============================================================================
@types('int', 'int', 'int[:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            # evaluation dependant uniquement de l'element
            for il_1 in range(0, degree+1):
                for il_2 in range(0, degree+1):
                            
                            for g1 in range(0, k1):
                               
                                    i_span_1 = spans[ie1, g1]
                                    i1 = i_span_1 - degree + il_1
                                    i2 = i_span_1 - degree + il_2
                                    
                                    
                                    bi_x = basis[ie1, il_1, 0, g1]
                                    bj_x = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    matrix[ degree+ i1, degree+ i2-i1]  += bi_x * bj_x * wvol
    # ...
    
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:,:]', 'int[:,:]', 'int[:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'real', 'real', 'int', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3, basis_1, basis_2, basis_3, weights_1, weights_2, weights_3, points_1, points_2, points_3, knots_1, knots_2, knots_3, vector_d, ovlp_value, S_DDM, domain_nb, rhs):
    from numpy import exp
    from numpy import pi
    from numpy import sin
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros 
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k3 = weights_3.shape[1]
    # ...
    lvalues_u = zeros((k1, k3))
    # ... build rhs
    for ie1 in range(0, ne1):
        for ie3 in range(0, ne3):

            lvalues_u[ : , : ] = 0.0
            for g1 in range(0, k1):
                for g3 in range(0, k3):

                    x    = points_1[ie1, g1]
                    y    = points_3[ie3, g3]

                    # ... test 0
                    f  = 2.*pi**2*sin(pi*x)*sin(pi*y)
                    lvalues_u[g1,g3] = f
            for il_1 in range(0, p1+1):
                for il_3 in range(0, p3+1):

                    v = 0.0
                    for g1 in range(0, k1):
                        for g3 in range(0, k3):
                            i_span_1 = spans_1[ie1, g1]
                            i_span_3 = spans_3[ie3, g3]

                            i1       = i_span_1 - p1 + il_1
                            i3       = i_span_3 - p3 + il_3                            
        
                            bi_0     = basis_1[ie1, il_1, 0, g1] * basis_3[ie3, il_3, 0, g3]

                            wvol     = weights_1[ie1, g1] * weights_3[ie3, g3]

                            u        = lvalues_u[g1,g3]
                           
                            rhs[i1+p1,i3+p3] += bi_0 * u * wvol

    # ...
    lcoeffs_d   = zeros((p2+1,p3+1))
    #---Computes All basis in at the interface of the overllape points
    nders          = 1
    degree         = p1
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis1         = zeros((nders+1, degree+1))
    xq   = ovlp_value
    #~~~~~~~~~~~~~~~
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
    span_1    = span
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
    # Multiply derivatives by correct factors
    r = degree
    ders[1,:] = ders[1,:] * r
    basis1[0,:] = ders[0,:]
    basis1[1,:] = ders[1,:]
    #---Computes All basis in at the interface of the overllape points
    nders          = 1
    degree         = p2
    #..
    basis2         = zeros((nders+1, degree+1))
    xq = ovlp_value
    #~~~~~~~~~~~~~~~
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
    span_2    = span
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
    # Multiply derivatives by correct factors
    r = degree
    ders[1,:] = ders[1,:] * r
    basis2[0,:] = ders[0,:]
    basis2[1,:] = ders[1,:]
    # ... build rhs
    if domain_nb == 0 :
       i1_ovrlp  = ne1+p1-1
       neum_sign = 1.
    else :                
       i1_ovrlp  = p1+1
       neum_sign = -1.
    for ie3 in range(0, ne3):
          for il_1 in range(0, p1+1):
              for il_3 in range(0, p3+1):

                    v = 0.0
                    for g3 in range(0, k3):

                          i_span_3 = spans_3[ie3, g3]
                          # ...
                          i3       = i_span_3 - p3 + il_3                            
                          i1       =   span_1 - p1 + il_1                          
                          # ...
                          lcoeffs_d[:,:] = vector_d[span_2 : span_2+p2+1, i_span_3 : i_span_3+p3+1]
                          # Assembles Roubin Condition
                          u   = 0.
                          ux  = 0.
                          for jl_2 in range(0, p2+1):
                             for jl_3 in range(0, p3+1):
                                bi_0      = basis2[0,jl_2] * basis_3[ie3, jl_3, 0, g3]
                                bi_x      = basis2[1,jl_2] * basis_3[ie3, jl_3, 0, g3]
                                # ...
                                coeff_d   = lcoeffs_d[jl_2, jl_3]
                                # ...
                                u        +=  coeff_d*bi_0
                                ux       +=  coeff_d*bi_x
                                
                          bi_0    = basis1[0,il_1] * basis_3[ie3, il_3, 0, g3]
                          # ...
                          wvol    = weights_3[ie3, g3]

                          rhs[i1+p1,i3+p3] += bi_0 * (neum_sign*ux+S_DDM*u) * wvol
    # ...


#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int[:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real', 'double[:]')
def assemble_vector_ex02(ne1, p1, spans_1,  basis_1,  weights_1, points_1,  knots_1, ovlp_value, rhs):
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1        = weights_1.shape[1]
    # ...
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
    # ... assemble the non vanishing point in the overlap value
    for il_1 in range(0, p1+1):
           rhs[p1+il_1]  = basis1[il_1]
    # ...    
    
#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:,:]', 'int[:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]','double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, rhs):

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
    # ...
    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        for ie2 in range(0, ne2):
            
            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]

                    # ... test 0
                    u   = sin(pi*x)*sin(pi*y)
                    ux  = pi*cos(pi*x)*sin(pi*y)
                    uy  = pi*sin(pi*x)*cos(pi*y) 
                    
                    i_span_1 = spans_1[ie1, g1]                            
                    i_span_2 = spans_2[ie2, g2]
                    lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
                    # ...
                    uh  = 0.0
                    uhx = 0.0
                    uhy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_1[ie1,il_1,0,g1] * basis_2[ie2,il_2,0,g2]
                              bj_x    = basis_1[ie1,il_1,1,g1] * basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1] * basis_2[ie2,il_2,1,g2]
 
                              # ...
                              coeff_u = lcoeffs_u[il_1,il_2]
                              # ...
                              uh     += coeff_u * bj_0
                              uhx    += coeff_u * bj_x
                              uhy    += coeff_u * bj_y

                    v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2+(uy-uhy)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1,p2] = norm_l2
    rhs[p1,p2+1] = norm_H1
    # ...
