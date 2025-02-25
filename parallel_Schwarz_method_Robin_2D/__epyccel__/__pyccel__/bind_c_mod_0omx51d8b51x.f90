module bind_c_mod_0omx51d8b51x

  use mod_0omx51d8b51x

  use, intrinsic :: ISO_C_Binding, only : C_F_Pointer , i64 => C_INT64_T &
        , c_ptr , f64 => C_DOUBLE
  implicit none

  contains

  !........................................
  subroutine bind_c_assemble_matrix_un_ex01(ne1, ne2, p1, p2, &
        bound_spans_1, bound_spans_1_shape_1, bound_spans_1_stride_1, &
        bound_spans_2, bound_spans_2_shape_1, bound_spans_2_stride_1, &
        bound_basis_1, bound_basis_1_shape_1, bound_basis_1_shape_2, &
        bound_basis_1_shape_3, bound_basis_1_shape_4, &
        bound_basis_1_stride_1, bound_basis_1_stride_2, &
        bound_basis_1_stride_3, bound_basis_1_stride_4, bound_basis_2, &
        bound_basis_2_shape_1, bound_basis_2_shape_2, &
        bound_basis_2_shape_3, bound_basis_2_shape_4, &
        bound_basis_2_stride_1, bound_basis_2_stride_2, &
        bound_basis_2_stride_3, bound_basis_2_stride_4, bound_weights_1 &
        , bound_weights_1_shape_1, bound_weights_1_shape_2, &
        bound_weights_1_stride_1, bound_weights_1_stride_2, &
        bound_weights_2, bound_weights_2_shape_1, &
        bound_weights_2_shape_2, bound_weights_2_stride_1, &
        bound_weights_2_stride_2, bound_points_1, &
        bound_points_1_shape_1, bound_points_1_shape_2, &
        bound_points_1_stride_1, bound_points_1_stride_2, &
        bound_points_2, bound_points_2_shape_1, bound_points_2_shape_2, &
        bound_points_2_stride_1, bound_points_2_stride_2, domain_nb, &
        S_DDM, ovlp_value, bound_matrix, bound_matrix_shape_1, &
        bound_matrix_shape_2, bound_matrix_shape_3, &
        bound_matrix_shape_4, bound_matrix_stride_1, &
        bound_matrix_stride_2, bound_matrix_stride_3, &
        bound_matrix_stride_4) bind(c)

    implicit none

    integer(i64), value :: ne1
    integer(i64), value :: ne2
    integer(i64), value :: p1
    integer(i64), value :: p2
    type(c_ptr), value :: bound_spans_1
    integer(i64), value :: bound_spans_1_shape_1
    integer(i64), value :: bound_spans_1_stride_1
    type(c_ptr), value :: bound_spans_2
    integer(i64), value :: bound_spans_2_shape_1
    integer(i64), value :: bound_spans_2_stride_1
    type(c_ptr), value :: bound_basis_1
    integer(i64), value :: bound_basis_1_shape_1
    integer(i64), value :: bound_basis_1_shape_2
    integer(i64), value :: bound_basis_1_shape_3
    integer(i64), value :: bound_basis_1_shape_4
    integer(i64), value :: bound_basis_1_stride_1
    integer(i64), value :: bound_basis_1_stride_2
    integer(i64), value :: bound_basis_1_stride_3
    integer(i64), value :: bound_basis_1_stride_4
    type(c_ptr), value :: bound_basis_2
    integer(i64), value :: bound_basis_2_shape_1
    integer(i64), value :: bound_basis_2_shape_2
    integer(i64), value :: bound_basis_2_shape_3
    integer(i64), value :: bound_basis_2_shape_4
    integer(i64), value :: bound_basis_2_stride_1
    integer(i64), value :: bound_basis_2_stride_2
    integer(i64), value :: bound_basis_2_stride_3
    integer(i64), value :: bound_basis_2_stride_4
    type(c_ptr), value :: bound_weights_1
    integer(i64), value :: bound_weights_1_shape_1
    integer(i64), value :: bound_weights_1_shape_2
    integer(i64), value :: bound_weights_1_stride_1
    integer(i64), value :: bound_weights_1_stride_2
    type(c_ptr), value :: bound_weights_2
    integer(i64), value :: bound_weights_2_shape_1
    integer(i64), value :: bound_weights_2_shape_2
    integer(i64), value :: bound_weights_2_stride_1
    integer(i64), value :: bound_weights_2_stride_2
    type(c_ptr), value :: bound_points_1
    integer(i64), value :: bound_points_1_shape_1
    integer(i64), value :: bound_points_1_shape_2
    integer(i64), value :: bound_points_1_stride_1
    integer(i64), value :: bound_points_1_stride_2
    type(c_ptr), value :: bound_points_2
    integer(i64), value :: bound_points_2_shape_1
    integer(i64), value :: bound_points_2_shape_2
    integer(i64), value :: bound_points_2_stride_1
    integer(i64), value :: bound_points_2_stride_2
    integer(i64), value :: domain_nb
    real(f64), value :: S_DDM
    real(f64), value :: ovlp_value
    type(c_ptr), value :: bound_matrix
    integer(i64), value :: bound_matrix_shape_1
    integer(i64), value :: bound_matrix_shape_2
    integer(i64), value :: bound_matrix_shape_3
    integer(i64), value :: bound_matrix_shape_4
    integer(i64), value :: bound_matrix_stride_1
    integer(i64), value :: bound_matrix_stride_2
    integer(i64), value :: bound_matrix_stride_3
    integer(i64), value :: bound_matrix_stride_4
    integer(i64), pointer :: spans_1(:)
    integer(i64), pointer :: spans_2(:)
    real(f64), pointer :: basis_1(:,:,:,:)
    real(f64), pointer :: basis_2(:,:,:,:)
    real(f64), pointer :: weights_1(:,:)
    real(f64), pointer :: weights_2(:,:)
    real(f64), pointer :: points_1(:,:)
    real(f64), pointer :: points_2(:,:)
    real(f64), pointer :: matrix(:,:,:,:)

    call C_F_Pointer(bound_spans_1, spans_1, [bound_spans_1_shape_1 * &
          bound_spans_1_stride_1])
    call C_F_Pointer(bound_spans_2, spans_2, [bound_spans_2_shape_1 * &
          bound_spans_2_stride_1])
    call C_F_Pointer(bound_basis_1, basis_1, [bound_basis_1_shape_4 * &
          bound_basis_1_stride_4,bound_basis_1_shape_3 * &
          bound_basis_1_stride_3,bound_basis_1_shape_2 * &
          bound_basis_1_stride_2,bound_basis_1_shape_1 * &
          bound_basis_1_stride_1])
    call C_F_Pointer(bound_basis_2, basis_2, [bound_basis_2_shape_4 * &
          bound_basis_2_stride_4,bound_basis_2_shape_3 * &
          bound_basis_2_stride_3,bound_basis_2_shape_2 * &
          bound_basis_2_stride_2,bound_basis_2_shape_1 * &
          bound_basis_2_stride_1])
    call C_F_Pointer(bound_weights_1, weights_1, [ &
          bound_weights_1_shape_2 * bound_weights_1_stride_2, &
          bound_weights_1_shape_1 * bound_weights_1_stride_1])
    call C_F_Pointer(bound_weights_2, weights_2, [ &
          bound_weights_2_shape_2 * bound_weights_2_stride_2, &
          bound_weights_2_shape_1 * bound_weights_2_stride_1])
    call C_F_Pointer(bound_points_1, points_1, [bound_points_1_shape_2 * &
          bound_points_1_stride_2,bound_points_1_shape_1 * &
          bound_points_1_stride_1])
    call C_F_Pointer(bound_points_2, points_2, [bound_points_2_shape_2 * &
          bound_points_2_stride_2,bound_points_2_shape_1 * &
          bound_points_2_stride_1])
    call C_F_Pointer(bound_matrix, matrix, [bound_matrix_shape_4 * &
          bound_matrix_stride_4,bound_matrix_shape_3 * &
          bound_matrix_stride_3,bound_matrix_shape_2 * &
          bound_matrix_stride_2,bound_matrix_shape_1 * &
          bound_matrix_stride_1])
    call assemble_matrix_un_ex01(ne1 = ne1, ne2 = ne2, p1 = p1, p2 = p2, &
          spans_1 = spans_1(1_i64::bound_spans_1_stride_1), spans_2 = &
          spans_2(1_i64::bound_spans_2_stride_1), basis_1 = basis_1( &
          1_i64::bound_basis_1_stride_4, 1_i64::bound_basis_1_stride_3, &
          1_i64::bound_basis_1_stride_2, 1_i64::bound_basis_1_stride_1) &
          , basis_2 = basis_2(1_i64::bound_basis_2_stride_4, 1_i64:: &
          bound_basis_2_stride_3, 1_i64::bound_basis_2_stride_2, 1_i64 &
          ::bound_basis_2_stride_1), weights_1 = weights_1(1_i64:: &
          bound_weights_1_stride_2, 1_i64::bound_weights_1_stride_1), &
          weights_2 = weights_2(1_i64::bound_weights_2_stride_2, 1_i64 &
          ::bound_weights_2_stride_1), points_1 = points_1(1_i64:: &
          bound_points_1_stride_2, 1_i64::bound_points_1_stride_1), &
          points_2 = points_2(1_i64::bound_points_2_stride_2, 1_i64:: &
          bound_points_2_stride_1), domain_nb = domain_nb, S_DDM = &
          S_DDM, ovlp_value = ovlp_value, matrix = matrix(1_i64:: &
          bound_matrix_stride_4, 1_i64::bound_matrix_stride_3, 1_i64:: &
          bound_matrix_stride_2, 1_i64::bound_matrix_stride_1))

  end subroutine bind_c_assemble_matrix_un_ex01
  !........................................

end module bind_c_mod_0omx51d8b51x
