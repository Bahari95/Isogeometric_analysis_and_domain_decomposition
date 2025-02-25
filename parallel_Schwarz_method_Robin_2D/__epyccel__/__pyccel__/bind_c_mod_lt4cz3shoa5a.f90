module bind_c_mod_lt4cz3shoa5a

  use mod_lt4cz3shoa5a

  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , c_ptr , &
        C_F_Pointer , f64 => C_DOUBLE
  implicit none

  contains

  !........................................
  subroutine bind_c_assemble_norm_ex01(ne1, ne2, p1, p2, bound_spans_1, &
        bound_spans_1_shape_1, bound_spans_1_stride_1, bound_spans_2, &
        bound_spans_2_shape_1, bound_spans_2_stride_1, bound_basis_1, &
        bound_basis_1_shape_1, bound_basis_1_shape_2, &
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
        bound_points_2_stride_1, bound_points_2_stride_2, &
        bound_vector_u, bound_vector_u_shape_1, bound_vector_u_shape_2, &
        bound_vector_u_stride_1, bound_vector_u_stride_2, bound_rhs, &
        bound_rhs_shape_1, bound_rhs_shape_2, bound_rhs_stride_1, &
        bound_rhs_stride_2) bind(c)

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
    type(c_ptr), value :: bound_vector_u
    integer(i64), value :: bound_vector_u_shape_1
    integer(i64), value :: bound_vector_u_shape_2
    integer(i64), value :: bound_vector_u_stride_1
    integer(i64), value :: bound_vector_u_stride_2
    type(c_ptr), value :: bound_rhs
    integer(i64), value :: bound_rhs_shape_1
    integer(i64), value :: bound_rhs_shape_2
    integer(i64), value :: bound_rhs_stride_1
    integer(i64), value :: bound_rhs_stride_2
    integer(i64), pointer :: spans_1(:)
    integer(i64), pointer :: spans_2(:)
    real(f64), pointer :: basis_1(:,:,:,:)
    real(f64), pointer :: basis_2(:,:,:,:)
    real(f64), pointer :: weights_1(:,:)
    real(f64), pointer :: weights_2(:,:)
    real(f64), pointer :: points_1(:,:)
    real(f64), pointer :: points_2(:,:)
    real(f64), pointer :: vector_u(:,:)
    real(f64), pointer :: rhs(:,:)

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
    call C_F_Pointer(bound_vector_u, vector_u, [bound_vector_u_shape_2 * &
          bound_vector_u_stride_2,bound_vector_u_shape_1 * &
          bound_vector_u_stride_1])
    call C_F_Pointer(bound_rhs, rhs, [bound_rhs_shape_2 * &
          bound_rhs_stride_2,bound_rhs_shape_1 * bound_rhs_stride_1])
    call assemble_norm_ex01(ne1 = ne1, ne2 = ne2, p1 = p1, p2 = p2, &
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
          bound_points_2_stride_1), vector_u = vector_u(1_i64:: &
          bound_vector_u_stride_2, 1_i64::bound_vector_u_stride_1), rhs &
          = rhs(1_i64::bound_rhs_stride_2, 1_i64::bound_rhs_stride_1))

  end subroutine bind_c_assemble_norm_ex01
  !........................................

end module bind_c_mod_lt4cz3shoa5a
