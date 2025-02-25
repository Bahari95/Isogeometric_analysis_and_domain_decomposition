module mod_kw6zyd5fqz8u


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
        C_DOUBLE
  implicit none

  contains

  !........................................
  subroutine assemble_matrix_un_ex01(ne1, ne2, p1, p2, spans_1, spans_2, &
        basis_1, basis_2, weights_1, weights_2, points_1, points_2, &
        domain_nb, S_DDM, ovlp_value, matrix)

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
    integer(i64), value :: domain_nb
    real(f64), value :: S_DDM
    real(f64), value :: ovlp_value
    real(f64), intent(inout) :: matrix(0:,0:,0:,0:)
    integer(i64) :: k1
    integer(i64) :: k2
    real(f64), allocatable :: lcoeffs_u(:,:)
    real(f64), allocatable :: arr_J_mat0(:,:)
    real(f64), allocatable :: arr_J_mat1(:,:)
    real(f64), allocatable :: arr_J_mat2(:,:)
    real(f64), allocatable :: arr_J_mat3(:,:)
    real(f64), allocatable :: J_mat(:,:)
    integer(i64) :: ie1
    integer(i64) :: i_span_1
    integer(i64) :: ie2
    integer(i64) :: i_span_2
    integer(i64) :: g1
    integer(i64) :: g2
    real(f64) :: x1
    real(f64) :: x2
    real(f64) :: F1x
    real(f64) :: F1y
    real(f64) :: F2y
    real(f64) :: F2x
    real(f64) :: det_Hess
    integer(i64) :: il_1
    integer(i64) :: il_2
    integer(i64) :: jl_1
    integer(i64) :: jl_2
    integer(i64) :: i1
    integer(i64) :: j1
    integer(i64) :: i2
    integer(i64) :: j2
    real(f64) :: v
    real(f64) :: bi_0
    real(f64) :: bj_0
    real(f64) :: bi_x1
    real(f64) :: bi_x2
    real(f64) :: bj_x1
    real(f64) :: bj_x2
    real(f64) :: bi_x
    real(f64) :: bi_y
    real(f64) :: bj_x
    real(f64) :: bj_y
    real(f64) :: wvol

    !... sizes
    k1 = size(weights_1, 1_i64, i64)
    k2 = size(weights_2, 1_i64, i64)
    !.. circle
    !...
    allocate(lcoeffs_u(0:p2 + 1_i64 - 1_i64, 0:p1 + 1_i64 - 1_i64))
    lcoeffs_u = 0.0_f64
    allocate(arr_J_mat0(0:k2 - 1_i64, 0:k1 - 1_i64))
    arr_J_mat0 = 0.0_f64
    allocate(arr_J_mat1(0:k2 - 1_i64, 0:k1 - 1_i64))
    arr_J_mat1 = 0.0_f64
    allocate(arr_J_mat2(0:k2 - 1_i64, 0:k1 - 1_i64))
    arr_J_mat2 = 0.0_f64
    allocate(arr_J_mat3(0:k2 - 1_i64, 0:k1 - 1_i64))
    arr_J_mat3 = 0.0_f64
    allocate(J_mat(0:k2 - 1_i64, 0:k1 - 1_i64))
    J_mat = 0.0_f64
    !... build matrices
    do ie1 = 0_i64, ne1 - 1_i64
      i_span_1 = spans_1(ie1)
      do ie2 = 0_i64, ne2 - 1_i64
        i_span_2 = spans_2(ie2)
        do g1 = 0_i64, k1 - 1_i64
          do g2 = 0_i64, k2 - 1_i64
            x1 = points_1(g1, ie1)
            x2 = points_2(g2, ie2)
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
            det_Hess = abs(F1x * F2y - F1y * F2x)
            !F=(F1, F2)
            arr_J_mat0(g2, g1) = F2y
            arr_J_mat1(g2, g1) = F1x
            arr_J_mat2(g2, g1) = F1y
            arr_J_mat3(g2, g1) = F2x
            J_mat(g2, g1) = det_Hess
          end do
        end do
        do il_1 = 0_i64, p1 + 1_i64 - 1_i64
          do il_2 = 0_i64, p2 + 1_i64 - 1_i64
            do jl_1 = 0_i64, p1 + 1_i64 - 1_i64
              do jl_2 = 0_i64, p2 + 1_i64 - 1_i64
                i1 = i_span_1 - p1 + il_1
                j1 = i_span_1 - p1 + jl_1
                i2 = i_span_2 - p2 + il_2
                j2 = i_span_2 - p2 + jl_2
                v = 0.0_f64
                do g1 = 0_i64, k1 - 1_i64
                  do g2 = 0_i64, k2 - 1_i64
                    bi_0 = basis_1(g1, 0_i64, il_1, ie1) * basis_2(g2, &
                          0_i64, il_2, ie2)
                    bj_0 = basis_1(g1, 0_i64, jl_1, ie1) * basis_2(g2, &
                          0_i64, jl_2, ie2)
                    bi_x1 = basis_1(g1, 1_i64, il_1, ie1) * basis_2(g2, &
                          0_i64, il_2, ie2)
                    bi_x2 = basis_1(g1, 0_i64, il_1, ie1) * basis_2(g2, &
                          1_i64, il_2, ie2)
                    bj_x1 = basis_1(g1, 1_i64, jl_1, ie1) * basis_2(g2, &
                          0_i64, jl_2, ie2)
                    bj_x2 = basis_1(g1, 0_i64, jl_1, ie1) * basis_2(g2, &
                          1_i64, jl_2, ie2)
                    bi_x = arr_J_mat0(g2, g1) * bi_x1 - arr_J_mat3(g2, &
                          g1) * bi_x2
                    bi_y = arr_J_mat1(g2, g1) * bi_x2 - arr_J_mat2(g2, &
                          g1) * bi_x1
                    bj_x = arr_J_mat0(g2, g1) * bj_x1 - arr_J_mat3(g2, &
                          g1) * bj_x2
                    bj_y = arr_J_mat1(g2, g1) * bj_x2 - arr_J_mat2(g2, &
                          g1) * bj_x1
                    wvol = weights_1(g1, ie1) * weights_2(g2, ie2)
                    v = v + (bi_x * bj_x + bi_y * bj_y) * wvol / J_mat( &
                          g2, g1)
                  end do
                end do
                matrix(p2 + j2 - i2, p1 + j1 - i1, p2 + i2, p1 + i1) = &
                      matrix(p2 + j2 - i2, p1 + j1 - i1, p2 + i2, p1 + &
                      i1) + v
              end do
            end do
          end do
        end do
      end do
    end do
    !... build matrices
    if (domain_nb == 0_i64) then
      ie1 = ne1 - 1_i64
      i_span_1 = spans_1(ie1)
      do ie2 = 0_i64, ne2 - 1_i64
        i_span_2 = spans_2(ie2)
        do g2 = 0_i64, k2 - 1_i64
          x1 = ovlp_value
          x2 = points_2(g2, ie2)
          F1y = (-(2.0_f64 * x1 - 1.0_f64)) * (2.0_f64 * x2 - 1.0_f64) / &
                sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x2 - 1.0_f64) ** &
                2_i64)
          F2y = 2.0_f64 * sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x1 - &
                1.0_f64) ** 2_i64)
          det_Hess = sqrt(F2y * F2y + F1y * F1y)
          J_mat(g2, 0_i64) = det_Hess
        end do
        do il_2 = 0_i64, p2 + 1_i64 - 1_i64
          do jl_2 = 0_i64, p2 + 1_i64 - 1_i64
            i2 = i_span_2 - p2 + il_2
            j2 = i_span_2 - p2 + jl_2
            v = 0.0_f64
            do g2 = 0_i64, k2 - 1_i64
              bi_0 = basis_2(g2, 0_i64, il_2, ie2)
              bj_0 = basis_2(g2, 0_i64, jl_2, ie2)
              wvol = weights_2(g2, ie2) * J_mat(g2, 0_i64)
              !...
              v = v + S_DDM * bi_0 * bj_0 * wvol
            end do
            matrix(p2 + j2 - i2, p1, p2 + i2, i_span_1 + p1) = matrix(p2 &
                  + j2 - i2, p1, p2 + i2, i_span_1 + p1) + v
          end do
        end do
      end do
    else
      ie1 = 0_i64
      i_span_1 = spans_1(ie1)
      do ie2 = 0_i64, ne2 - 1_i64
        i_span_2 = spans_2(ie2)
        do g2 = 0_i64, k2 - 1_i64
          x1 = ovlp_value
          x2 = points_2(g2, ie2)
          F1y = (-(2.0_f64 * x1 - 1.0_f64)) * (2.0_f64 * x2 - 1.0_f64) / &
                sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x2 - 1.0_f64) ** &
                2_i64)
          F2y = 2.0_f64 * sqrt(1.0_f64 - 0.5_f64 * (2.0_f64 * x1 - &
                1.0_f64) ** 2_i64)
          det_Hess = sqrt(F2y * F2y + F1y * F1y)
          J_mat(g2, 0_i64) = det_Hess
        end do
        do il_2 = 0_i64, p2 + 1_i64 - 1_i64
          do jl_2 = 0_i64, p2 + 1_i64 - 1_i64
            i2 = i_span_2 - p2 + il_2
            j2 = i_span_2 - p2 + jl_2
            v = 0.0_f64
            do g2 = 0_i64, k2 - 1_i64
              bi_0 = basis_2(g2, 0_i64, il_2, ie2)
              bj_0 = basis_2(g2, 0_i64, jl_2, ie2)
              wvol = weights_2(g2, ie2) * J_mat(g2, 0_i64)
              !...
              !...
              v = v + S_DDM * bi_0 * bj_0 * wvol
            end do
            matrix(p2 + j2 - i2, p1, p2 + i2, p1) = matrix(p2 + j2 - i2, &
                  p1, p2 + i2, p1) + v
          end do
        end do
      end do
    end if
    !...
    if (allocated(arr_J_mat3)) then
      deallocate(arr_J_mat3)
    end if
    if (allocated(arr_J_mat1)) then
      deallocate(arr_J_mat1)
    end if
    if (allocated(J_mat)) then
      deallocate(J_mat)
    end if
    if (allocated(arr_J_mat2)) then
      deallocate(arr_J_mat2)
    end if
    if (allocated(arr_J_mat0)) then
      deallocate(arr_J_mat0)
    end if
    if (allocated(lcoeffs_u)) then
      deallocate(lcoeffs_u)
    end if

  end subroutine assemble_matrix_un_ex01
  !........................................

end module mod_kw6zyd5fqz8u
