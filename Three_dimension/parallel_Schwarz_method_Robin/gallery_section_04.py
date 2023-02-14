__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex02',
           'assemble_norm_ex02'
]

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
    
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex11(ne, degree, spans, basis, weights, points,  matrix):

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
                                    bj_x = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v
    # ...
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex12(ne, degree, spans, basis, weights, points,  matrix):
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
                                
                                    bi_x = basis[ie1, il_1, 0, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v
    # ...    
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

#==============================================================================
# .. in uniform mesh Matrix
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        matrix):

    # ... sizes
    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1  = points_1[ie1,g1]
                    x2  = points_2[ie2,g2]

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

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]


                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                                    # ...
                                    v += (bj_x1* bi_x1 + bj_x2 * bi_x2) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:,:]', 'real', 'real', 'int', 'double[:,:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, knots_1, knots_2, knots_3, knots_4, knots_5, knots_6, vector_d, ovlp_value, S_DDM, domain_nb, rhs):

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
    k3        = weights_3.shape[1]

    # ...
    lcoeffs_d   = zeros((p1+1,p2+1,p3+1))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            for ie3 in range(0, ne3):
                i_span_3 = spans_3[ie3]

                for il_1 in range(0, p1+1):
                  for il_2 in range(0, p2+1):
                    for il_3 in range(0, p3+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2
                      i3 = i_span_3 - p3 + il_3

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):
                            for g3 in range(0, k3):

                              bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                              #..
                              wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]
                              #++
                              t  = points_1[ie1, g1]
                              x  = points_2[ie2, g2]
                              y  = points_3[ie3, g3]
                              #.. Test 0
                              u    = 3*pi**2*sin(pi*t)*sin(pi*x)*sin(pi*y)
                              #.. Test 0
                              #st = t
                              #t  = x
                              #x  = st
                              #u    =  51.0*pi**2*x**2*y*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) + 6*x**2*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) + 48.0*pi*x*y*(1.0 - y)*sin(pi*t)*cos(pi*(4.0 - 4.0*x)) - 6*y*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x))
                                                            
                              v   += bi_0 * u * wvol

                      rhs[i1+p1,i2+p2,i3+p3] += v   

    lvalues_u      = zeros((k2, k3))
    lvalues_ux     = zeros((k2, k3))
    #---Computes All basis in a new points
    nders          = 1
    degree         = p4
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis4         = zeros((nders+1, degree+1))
    for i in range(1):
            #span = find_span( knots, degree, xq )
            xq = ovlp_value
            #~~~~~~~~~~~~~~~
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_4)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_4[low ]: 
                 span = low
            if xq >= knots_4[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_4[span] or xq >= knots_4[span+1]:
                 if xq < knots_4[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_4[span-j]
                right[j] = knots_4[span+1+j] - xq
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
            basis4[0,:] = ders[0,:]
            basis4[1,:] = ders[1,:]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    if domain_nb == 0 :
      i1_ovrlp  = ne1+2*p1-1
      neum_sign = 1.
    else :
      i1_ovrlp  = p1
      neum_sign = -1.
                
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]
        for ie3 in range(0, ne3):
           i_span_3 = spans_3[ie3]
           
           for g2 in range(0, k2):
               for g3 in range(0, k3):
                    #... We compute firstly the span in new adapted points                             
                    #...                    
                    xq        = ovlp_value
                    degree    = p4
                    low       = degree
                    high      = len(knots_4)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_4[low ]: 
                         span = low
                    if xq >= knots_4[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_4[span] or xq >= knots_4[span+1]:
                         if xq < knots_4[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_4    = span 
                    # ...
                    lcoeffs_d[:, : , :] = vector_d[span_4 : span_4+p4+1, i_span_2 : i_span_2+p5+1, i_span_3 : i_span_3+p6+1]
                    # ...
                    u  = 0.
                    ux  = 0.
                    for il_1 in range(0, p4+1):
                       for il_2 in range(0, p5+1):
                          for il_3 in range(0, p6+1):
                             bi_0    = basis4[0,il_1] * basis_5[ie2, il_2, 0, g2] * basis_6[ie3, il_3, 0, g3]
                             bi_x    = basis4[1,il_1] * basis_5[ie2, il_2, 0, g2] * basis_6[ie3, il_3, 0, g3]
                             # ...
                             coeff_d = lcoeffs_d[il_1, il_2, il_3]
                             # ...
                             u      +=  coeff_d*bi_0
                             ux     +=  coeff_d*bi_x
                    lvalues_u[g2, g3]  = u
                    lvalues_ux[g2, g3] = ux
                                                
           for il_2 in range(0, p2+1):
              i2 = i_span_2 - p2 + il_2
              for il_3 in range(0, p3+1):
                 i3 = i_span_3 - p3 + il_3

                 v = 0.0
                 for g2 in range(0, k2):
                    for g3 in range(0, k3):
                       bi_0     =  basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                       wsurf    =  weights_2[ie2, g2] * weights_3[ie3, g3]
                  
                       #.. 
                       v    += bi_0 * (neum_sign*lvalues_ux[g2, g3]+S_DDM*lvalues_u[g2, g3]) * wsurf
                      
                 rhs[i1_ovrlp,i2+p2,i3+p3] += v
    # ...


#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:,:]', 'real', 'double[:,:]')
def assemble_vector_ex02(ne1, ne2, ne3, ne4, ne5, p1, p2, p3, p4, p5, spans_1, spans_2,  spans_3, spans_4, spans_5, basis_1, basis_2, basis_3, basis_4, basis_5, weights_1, weights_2, weights_3, weights_4, weights_5, points_1, points_2, points_3, points_4, points_5, knots_1, knots_2, knots_3, knots_4, knots_5, vector_d, value, rhs):

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
    lcoeffs_d   = zeros((p5+1,p3+1,p4+1))
    # ..
    lvalues_u   = zeros((k1, k2))
    #---Computes All basis in a new points
    nders          = 2
    degree         = p3
    #..
    ne, nq         = points_1.shape
    xx             = zeros(nq)

    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    basis3         = zeros( (ne, degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = points_1[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_3)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_3[low ]: 
                 span = low
            if xq >= knots_3[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_3[span] or xq >= knots_3[span+1]:
                 if xq < knots_3[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_3[span-j]
                right[j] = knots_3[span+1+j] - xq
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
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis3[ie,:,0,iq] = ders[0,:]
            basis3[ie,:,1,iq] = ders[1,:]
            basis3[ie,:,2,iq] = ders[2,:]

    degree         = p4
    #...
    basis4         = zeros( (ne,degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = points_2[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_4)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_4[low ]: 
                 span = low
            if xq >= knots_4[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_4[span] or xq >= knots_4[span+1]:
                 if xq < knots_4[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_4[span-j]
                right[j] = knots_4[span+1+j] - xq
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
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis4[ie,:,0,iq] = ders[0,:]
            basis4[ie,:,1,iq] = ders[1,:]
            basis4[ie,:,2,iq] = ders[2,:]
    degree         = p5
    #...
    basis5         = zeros( (1,degree+1, nders+1, nq))
    for ie in range(1):
        xx[:] = value
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_5)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_5[low ]: 
                 span = low
            if xq >= knots_5[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_5[span] or xq >= knots_5[span+1]:
                 if xq < knots_5[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_5[span-j]
                right[j] = knots_5[span+1+j] - xq
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
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis5[ie,:,0,iq] = ders[0,:]
            basis5[ie,:,1,iq] = ders[1,:]
            basis5[ie,:,2,iq] = ders[2,:]            
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    #... We compute firstly the span in new adapted points
                    xq        = points_1[ie1, g1]
                    degree    = p3
                    low       = degree
                    high      = len(knots_3)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_3[low ]: 
                         span = low
                    if xq >= knots_3[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_3[span] or xq >= knots_3[span+1]:
                         if xq < knots_3[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_3    = span
                    #...                    
                    xq        = points_2[ie2, g2]
                    degree    = p4
                    low       = degree
                    high      = len(knots_4)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_4[low ]: 
                         span = low
                    if xq >= knots_4[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_4[span] or xq >= knots_4[span+1]:
                         if xq < knots_4[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_4    = span                                  
                    #...                    
                    xq        = value
                    degree    = p5
                    low       = degree
                    high      = len(knots_5)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_5[low ]: 
                         span = low
                    if xq >= knots_5[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_5[span] or xq >= knots_5[span+1]:
                         if xq < knots_5[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_5    = span 
                    # ...
                    lcoeffs_d[:, : , :] = vector_d[span_5 : span_5+p5+1, i_span_1 : i_span_1+p3+1, i_span_2 : i_span_2+p4+1]
                    # ...
                    u  = 0.
                    for il_3 in range(0, p5+1):
                       for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                             bi_0    = basis5[0, il_3, 0, 0] * basis_3[ie1, il_1, 0, g1] * basis_4[ie2, il_2, 0, g2]
                             # ...
                             coeff_d = lcoeffs_d[il_3, il_1, il_2]
                             # ...
                             u      +=  coeff_d*bi_0
                    lvalues_u[g1, g2] = u

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):

                              bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                              # ...
                              wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                              # ...
                              u    = lvalues_u[g1, g2]
                              # ...        
                              v   += bi_0 * u * wvol

                      rhs[i1+p1,i2+p2] += v   
    # ...    
    

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:,:]', 'real', 'double[:,:]')
def assemble_vector_ex03(ne1, ne2, ne3, ne4, ne5, p1, p2, p3, p4, p5, spans_1, spans_2,  spans_3, spans_4, spans_5, basis_1, basis_2, basis_3, basis_4, basis_5, weights_1, weights_2, weights_3, weights_4, weights_5, points_1, points_2, points_3, points_4, points_5, knots_1, knots_2, knots_3, knots_4, knots_5, vector_d, ovlp_value, rhs):

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
    lcoeffs_d   = zeros((p5+1,p3+1,p4+1))
    # ..
    lvalues_u   = zeros((k1, k2))
    #---Computes All basis in a new points
    nders          = 0
    degree         = p5
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis5         = zeros(degree+1)
    for i in range(1):
            #span = find_span( knots, degree, xq )
            xq = ovlp_value
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_5)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_5[low ]: 
                 span = low
            if xq >= knots_5[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_5[span] or xq >= knots_5[span+1]:
                 if xq < knots_5[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_5[span-j]
                right[j] = knots_5[span+1+j] - xq
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
            basis5[:] = ders[0,:]
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    #... We compute firstly the span in new adapted points                             
                    #...                    
                    xq        = ovlp_value
                    degree    = p5
                    low       = degree
                    high      = len(knots_5)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_5[low ]: 
                         span = low
                    if xq >= knots_5[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_5[span] or xq >= knots_5[span+1]:
                         if xq < knots_5[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_5    = span 
                    # ...
                    lcoeffs_d[:, : , :] = vector_d[span_5 : span_5+p5+1, i_span_1 : i_span_1+p3+1, i_span_2 : i_span_2+p4+1]
                    # ...
                    u  = 0.
                    for il_3 in range(0, p5+1):
                       for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                             bi_0    = basis5[il_3] * basis_3[ie1, il_1, 0, g1] * basis_4[ie2, il_2, 0, g2]
                             # ...
                             coeff_d = lcoeffs_d[il_3, il_1, il_2]
                             # ...
                             u      +=  coeff_d*bi_0
                    lvalues_u[g1, g2] = u

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):

                              bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                              # ...
                              wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                              # ...
                              u    = lvalues_u[g1, g2]
                              # ...        
                              v   += bi_0 * u * wvol

                      rhs[i1+p1,i2+p2] += v   
    # ...    

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:,:]', 'real', 'double[:,:]')
def assemble_vector_ex04(ne1, ne2, ne3, ne4, ne5, p1, p2, p3, p4, p5, spans_1, spans_2,  spans_3, spans_4, spans_5, basis_1, basis_2, basis_3, basis_4, basis_5, weights_1, weights_2, weights_3, weights_4, weights_5, points_1, points_2, points_3, points_4, points_5, knots_1, knots_2, knots_3, knots_4, knots_5, vector_d, ovlp_value, rhs):

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
    lcoeffs_d   = zeros((p5+1,p3+1,p4+1))
    # ..
    lvalues_u   = zeros((k1, k2))
    #---Computes All basis in a new points
    nders          = 0
    degree         = p5
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis5         = zeros(degree+1)
    for i in range(1):
            #span = find_span( knots, degree, xq )
            xq = ovlp_value
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_5)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_5[low ]: 
                 span = low
            if xq >= knots_5[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_5[span] or xq >= knots_5[span+1]:
                 if xq < knots_5[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_5[span-j]
                right[j] = knots_5[span+1+j] - xq
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
            basis5[:] = ders[0,:]
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    #... We compute firstly the span in new adapted points                             
                    #...                    
                    xq        = ovlp_value
                    degree    = p5
                    low       = degree
                    high      = len(knots_5)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_5[low ]: 
                         span = low
                    if xq >= knots_5[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_5[span] or xq >= knots_5[span+1]:
                         if xq < knots_5[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_5    = span 
                    # ...
                    lcoeffs_d[:, : , :] = vector_d[span_5 : span_5+p5+1, i_span_1 : i_span_1+p3+1, i_span_2 : i_span_2+p4+1]
                    # ...
                    u  = 0.
                    for il_3 in range(0, p5+1):
                       for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                             bi_0    = basis5[il_3] * basis_3[ie1, il_1, 0, g1] * basis_4[ie2, il_2, 0, g2]
                             # ...
                             coeff_d = lcoeffs_d[il_3, il_1, il_2]
                             # ...
                             u      +=  coeff_d*bi_0
                    lvalues_u[g1, g2] = u

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):

                              bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                              # ...
                              wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                              # ...
                              u    = lvalues_u[g1, g2]
                              # ...        
                              v   += bi_0 * u * wvol

                      rhs[i1+p1,i2+p2] += v   
    # ...    
    
#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:]', 'double[:,:,:]')
def assemble_norm_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, vector_u, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1         = weights_1.shape[1]
    k2         = weights_2.shape[1]
    k3         = weights_3.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1,p3+1))
    # ...
    lvalues_u  = zeros((k1, k2, k3))
    lvalues_ut = zeros((k1, k2, k3))
    lvalues_ux = zeros((k1, k2, k3))
    lvalues_uy = zeros((k1, k2, k3))

    # ...
    norm_H1    = 0.
    norm_l2    = 0.
    for ie1 in range(0, ne1):
       i_span_1 = spans_1[ie1]
       for ie2 in range(0, ne2):
          i_span_2 = spans_2[ie2]
          for ie3 in range(0, ne3):
             i_span_3 = spans_3[ie3]

             lvalues_u[ : , : , :]  = 0.0
             lvalues_ut[ : , : , :] = 0.0
             lvalues_ux[ : , : , :] = 0.0
             lvalues_uy[ : , : , :] = 0.0
             lcoeffs_u[ : , : , :]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
             for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                   for il_3 in range(0, p3+1):
                      coeff_u = lcoeffs_u[il_1,il_2,il_3]

                      for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]
                            for g3 in range(0, k3):
                               b3  = basis_3[ie3,il_3,0,g3]
                               db3 = basis_3[ie3,il_3,1,g3]

                               lvalues_u[g1,g2,g3]  += coeff_u*b1*b2*b3
                               lvalues_ut[g1,g2,g3] += coeff_u*db1*b2*b3
                               lvalues_ux[g1,g2,g3] += coeff_u*b1*db2*b3
                               lvalues_uy[g1,g2,g3] += coeff_u*b1*b2*db3

             v = 0.0
             w = 0.0
             for g1 in range(0, k1):
               for g2 in range(0, k2):
                 for g3 in range(0, k3):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]

                    t  = points_1[ie1, g1]
                    x  = points_2[ie2, g2]
                    y  = points_3[ie3, g3]

                    # ... TEST 1 
                    u    = sin(pi*t)*sin(pi*x)*sin(pi*y)
                    ut   = pi*cos(pi*t)*sin(pi*x)*sin(pi*y)
                    ux   = pi*sin(pi*t)*cos(pi*x)*sin(pi*y)
                    uy   = pi*sin(pi*t)*sin(pi*x)*cos(pi*y)
                    # ... Test 2 
                    #st = t
                    #t  = x
                    #x  = st
                    #u    =  sin(pi*t)*x**2*y*3*sin(4.*pi*(1.-x))*(1.-y)
                    #ux   =  3*pi*x**2*y*(1.0 - y)*sin(pi*(4.0 - 4.0*x))*cos(pi*t) 
                    #ut   = -12.0*pi*x**2*y*(1.0 - y)*sin(pi*t)*cos(pi*(4.0 - 4.0*x)) + 6*x*y*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x))
                    #uy   = -3*x**2*y*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) + 3*x**2*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) 
                    
                    uh  = lvalues_u[g1,g2,g3]
                    uht = lvalues_ut[g1,g2,g3]
                    uhx = lvalues_ux[g1,g2,g3]
                    uhy = lvalues_uy[g1,g2,g3]

                    v  += ((ut-uht)**2+(ux-uhx)**2+(uy-uhy)**2) * wvol
                    w  += (u-uh)**2 * wvol

             norm_H1 += v
             norm_l2 += w

    norm_H1 = sqrt(norm_H1)
    norm_l2 = sqrt(norm_l2)

    rhs[p1,p2,p3]   = norm_l2
    rhs[p1,p2,p3+1] = norm_H1
    # ...
