module mod_i3hhw1bxhl9q


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
        C_DOUBLE
  implicit none

  contains

  !........................................
  subroutine assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2, &
        basis_1, basis_2, weights_1, weights_2, points_1, points_2, &
        vector_u, rhs)

    implicit none

    integer(i64), value :: ne1
    integer(i64), value :: ne2
    integer(i64), value :: p1
    integer(i64), value :: p2
    integer(i64), intent(in) :: spans_1(0:)
    integer(i64), intent(in) :: spans_2(0:)
    real(f64), intent(in) :: basis_1(0:,0:,0:,0:)
    real(f64), intent(in) :: basis_2(0:,0:,0:,0:)
    real(f64), intent(in) :: weights_1(0:,0:)
    real(f64), intent(in) :: weights_2(0:,0:)
    real(f64), intent(in) :: points_1(0:,0:)
    real(f64), intent(in) :: points_2(0:,0:)
    real(f64), intent(in) :: vector_u(0:,0:)
    real(f64), intent(inout) :: rhs(0:,0:)
    integer(i64) :: k1
    integer(i64) :: k2
    real(f64), allocatable :: lcoeffs_u(:,:)
    real(f64), allocatable :: lvalues_ux(:,:)
    real(f64), allocatable :: lvalues_uy(:,:)
    real(f64), allocatable :: lvalues_u(:,:)
    real(f64) :: error_l2
    real(f64) :: error_H1
    integer(i64) :: ie1
    integer(i64) :: i_span_1
    integer(i64) :: ie2
    integer(i64) :: i_span_2
    integer(i64) :: il_1
    integer(i64) :: il_2
    real(f64) :: coeff_u
    integer(i64) :: g1
    real(f64) :: b1
    real(f64) :: db1
    integer(i64) :: g2
    real(f64) :: b2
    real(f64) :: db2
    real(f64) :: w
    real(f64) :: v
    real(f64) :: wvol
    real(f64) :: x1
    real(f64) :: x2
    real(f64) :: uh
    real(f64) :: sx
    real(f64) :: sy
    real(f64) :: x
    real(f64) :: y
    real(f64) :: F1x
    real(f64) :: F1y
    real(f64) :: F2y
    real(f64) :: F2x
    real(f64) :: det_J
    real(f64) :: f
    real(f64) :: fx
    real(f64) :: fy
    real(f64) :: uhx
    real(f64) :: uhy

    !... sizes
    k1 = size(weights_1, 1_i64, i64)
    k2 = size(weights_2, 1_i64, i64)
    !.. circle
    !...
    allocate(lcoeffs_u(0:p2 + 1_i64 - 1_i64, 0:p1 + 1_i64 - 1_i64))
    lcoeffs_u = 0.0_f64
    allocate(lvalues_ux(0:k2 - 1_i64, 0:k1 - 1_i64))
    lvalues_ux = 0.0_f64
    allocate(lvalues_uy(0:k2 - 1_i64, 0:k1 - 1_i64))
    lvalues_uy = 0.0_f64
    allocate(lvalues_u(0:k2 - 1_i64, 0:k1 - 1_i64))
    lvalues_u = 0.0_f64
    error_l2 = 0.0_f64
    error_H1 = 0.0_f64
    !...
    do ie1 = 0_i64, ne1 - 1_i64
      i_span_1 = spans_1(ie1)
      do ie2 = 0_i64, ne2 - 1_i64
        i_span_2 = spans_2(ie2)
        lvalues_u(:, :) = 0.0_f64
        lvalues_ux(:, :) = 0.0_f64
        lvalues_uy(:, :) = 0.0_f64
        lcoeffs_u(:, :) = vector_u(i_span_2:i_span_2 + p2 + 1_i64 - &
              1_i64, i_span_1:i_span_1 + p1 + 1_i64 - 1_i64)
        do il_1 = 0_i64, p1 + 1_i64 - 1_i64
          do il_2 = 0_i64, p2 + 1_i64 - 1_i64
            coeff_u = lcoeffs_u(il_2, il_1)
            do g1 = 0_i64, k1 - 1_i64
              b1 = basis_1(g1, 0_i64, il_1, ie1)
              db1 = basis_1(g1, 1_i64, il_1, ie1)
              do g2 = 0_i64, k2 - 1_i64
                b2 = basis_2(g2, 0_i64, il_2, ie2)
                db2 = basis_2(g2, 1_i64, il_2, ie2)
                lvalues_u(g2, g1) = lvalues_u(g2, g1) + coeff_u * b1 * &
                      b2
                lvalues_ux(g2, g1) = lvalues_ux(g2, g1) + coeff_u * db1 &
                      * b2
                lvalues_uy(g2, g1) = lvalues_uy(g2, g1) + coeff_u * b1 * &
                      db2
              end do
            end do
          end do
        end do
        w = 0.0_f64
        v = 0.0_f64
        do g1 = 0_i64, k1 - 1_i64
          do g2 = 0_i64, k2 - 1_i64
            wvol = weights_1(g1, ie1) * weights_2(g2, ie2)
            x1 = points_1(g1, ie1)
            x2 = points_2(g2, ie2)
            uh = lvalues_u(g2, g1)
            sx = lvalues_ux(g2, g1)
            sy = lvalues_uy(g2, g1)
            x = (2.0_f64 * x1 - 1.0_f64) * sqrt(1.0_f64 - 0.5_f64 * ( &
                  2.0_f64 * x2 - 1.0_f64) ** 2_i64)
            y = (2.0_f64 * x2 - 1.0_f64) * sqrt(1.0_f64 - 0.5_f64 * ( &
                  2.0_f64 * x1 - 1.0_f64) ** 2_i64)
            F1x = 2.0_f64 * sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x2 - &
                  1.0_f64) ** 2_i64)
            F1y = (-(2.0_f64 * x1 - 1.0_f64)) * (2.0_f64 * x2 - 1.0_f64 &
                  ) / sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x2 - 1.0_f64 &
                  ) ** 2_i64)
            F2y = 2.0_f64 * sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x1 - &
                  1.0_f64) ** 2_i64)
            F2x = (-(2.0_f64 * x1 - 1.0_f64)) * (2.0_f64 * x2 - 1.0_f64 &
                  ) / sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x1 - 1.0_f64 &
                  ) ** 2_i64)
            det_J = abs(F1x * F2y - F1y * F2x)
            !... TEST 1
            !f   = exp(-500*(x**2 + y**2 - 0.2)**2)
            !fx  = -2000*x*(x**2 + y**2 - 0.2)*exp(-500*(x**2 + y**2 - 0.2)**2)
            !fy  = -2000*y*(x**2 + y**2 - 0.2)*exp(-500*(x**2 + y**2 - 0.2)**2)
            !... TEST 2
            f = 1.0_f64 - x * x - y * y
            fx = (-2_i64) * x
            fy = (-2_i64) * y
            uhx = (F2y * sx - F2x * sy) / det_J
            uhy = (F1x * sy - F1y * sx) / det_J
            w = w + ((uhx - fx) ** 2_i64 + (uhy - fy) ** 2_i64) * wvol * &
                  det_J
            v = v + (uh - f) ** 2_i64 * wvol * det_J
          end do
        end do
        error_H1 = error_H1 + w
        error_l2 = error_l2 + v
      end do
    end do
    rhs(p2, p1) = sqrt(error_l2)
    rhs(p2 + 1_i64, p1) = sqrt(error_H1)
    !...
    if (allocated(lvalues_ux)) then
      deallocate(lvalues_ux)
    end if
    if (allocated(lvalues_u)) then
      deallocate(lvalues_u)
    end if
    if (allocated(lvalues_uy)) then
      deallocate(lvalues_uy)
    end if
    if (allocated(lcoeffs_u)) then
      deallocate(lcoeffs_u)
    end if

  end subroutine assemble_norm_ex01
  !........................................

end module mod_i3hhw1bxhl9q
