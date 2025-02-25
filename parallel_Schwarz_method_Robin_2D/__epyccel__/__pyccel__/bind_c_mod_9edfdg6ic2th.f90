module bind_c_mod_9edfdg6ic2th

  use mod_9edfdg6ic2th

  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , c_ptr , i64 &
        => C_INT64_T , C_F_Pointer
  implicit none

  contains

  !........................................
  subroutine bind_c_assemble_vector_ex01(ne1, ne2, ne3, ne4, p1, p2, p3, &
        p4, bound_spans_1, bound_spans_1_shape_1, &
        bound_spans_1_stride_1, bound_spans_2, bound_spans_2_shape_1, &
        bound_spans_2_stride_1, bound_spans_3, bound_spans_3_shape_1, &
        bound_spans_3_stride_1, bound_spans_4, bound_spans_4_shape_1, &
        bound_spans_4_stride_1, bound_basis_1, bound_basis_1_shape_1, &
        bound_basis_1_shape_2, bound_basis_1_shape_3, &
        bound_basis_1_shape_4, bound_basis_1_stride_1, &
        bound_basis_1_stride_2, bound_basis_1_stride_3, &
        bound_basis_1_stride_4, bound_basis_2, bound_basis_2_shape_1, &
        bound_basis_2_shape_2, bound_basis_2_shape_3, &
        bound_basis_2_shape_4, bound_basis_2_stride_1, &
        bound_basis_2_stride_2, bound_basis_2_stride_3, &
        bound_basis_2_stride_4, bound_basis_3, bound_basis_3_shape_1, &
        bound_basis_3_shape_2, bound_basis_3_shape_3, &
        bound_basis_3_shape_4, bound_basis_3_stride_1, &
        bound_basis_3_stride_2, bound_basis_3_stride_3, &
        bound_basis_3_stride_4, bound_basis_4, bound_basis_4_shape_1, &
        bound_basis_4_shape_2, bound_basis_4_shape_3, &
        bound_basis_4_shape_4, bound_basis_4_stride_1, &
        bound_basis_4_stride_2, bound_basis_4_stride_3, &
        bound_basis_4_stride_4, bound_weights_1, &
        bound_weights_1_shape_1, bound_weights_1_shape_2, &
        bound_weights_1_stride_1, bound_weights_1_stride_2, &
        bound_weights_2, bound_weights_2_shape_1, &
        bound_weights_2_shape_2, bound_weights_2_stride_1, &
        bound_weights_2_stride_2, bound_weights_3, &
        bound_weights_3_shape_1, bound_weights_3_shape_2, &
        bound_weights_3_stride_1, bound_weights_3_stride_2, &
        bound_weights_4, bound_weights_4_shape_1, &
        bound_weights_4_shape_2, bound_weights_4_stride_1, &
        bound_weights_4_stride_2, bound_points_1, &
        bound_points_1_shape_1, bound_points_1_shape_2, &
        bound_points_1_stride_1, bound_points_1_stride_2, &
        bound_points_2, bound_points_2_shape_1, bound_points_2_shape_2, &
        bound_points_2_stride_1, bound_points_2_stride_2, &
        bound_points_3, bound_points_3_shape_1, bound_points_3_shape_2, &
        bound_points_3_stride_1, bound_points_3_stride_2, &
        bound_points_4, bound_points_4_shape_1, bound_points_4_shape_2, &
        bound_points_4_stride_1, bound_points_4_stride_2, bound_knots_1 &
        , bound_knots_1_shape_1, bound_knots_1_stride_1, bound_knots_2, &
        bound_knots_2_shape_1, bound_knots_2_stride_1, bound_knots_3, &
        bound_knots_3_shape_1, bound_knots_3_stride_1, bound_knots_4, &
        bound_knots_4_shape_1, bound_knots_4_stride_1, bound_vector_d, &
        bound_vector_d_shape_1, bound_vector_d_shape_2, &
        bound_vector_d_stride_1, bound_vector_d_stride_2, ovlp_value, &
        S_DDM, domain_nb, bound_rhs, bound_rhs_shape_1, &
        bound_rhs_shape_2, bound_rhs_stride_1, bound_rhs_stride_2) bind &
        (c)

    implicit none

    integer(i64), value :: ne1
    integer(i64), value :: ne2
    integer(i64), value :: ne3
    integer(i64), value :: ne4
    integer(i64), value :: p1
    integer(i64), value :: p2
    integer(i64), value :: p3
    integer(i64), value :: p4
    type(c_ptr), value :: bound_spans_1
    integer(i64), value :: bound_spans_1_shape_1
    integer(i64), value :: bound_spans_1_stride_1
    type(c_ptr), value :: bound_spans_2
    integer(i64), value :: bound_spans_2_shape_1
    integer(i64), value :: bound_spans_2_stride_1
    type(c_ptr), value :: bound_spans_3
    integer(i64), value :: bound_spans_3_shape_1
    integer(i64), value :: bound_spans_3_stride_1
    type(c_ptr), value :: bound_spans_4
    integer(i64), value :: bound_spans_4_shape_1
    integer(i64), value :: bound_spans_4_stride_1
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
    type(c_ptr), value :: bound_basis_3
    integer(i64), value :: bound_basis_3_shape_1
    integer(i64), value :: bound_basis_3_shape_2
    integer(i64), value :: bound_basis_3_shape_3
    integer(i64), value :: bound_basis_3_shape_4
    integer(i64), value :: bound_basis_3_stride_1
    integer(i64), value :: bound_basis_3_stride_2
    integer(i64), value :: bound_basis_3_stride_3
    integer(i64), value :: bound_basis_3_stride_4
    type(c_ptr), value :: bound_basis_4
    integer(i64), value :: bound_basis_4_shape_1
    integer(i64), value :: bound_basis_4_shape_2
    integer(i64), value :: bound_basis_4_shape_3
    integer(i64), value :: bound_basis_4_shape_4
    integer(i64), value :: bound_basis_4_stride_1
    integer(i64), value :: bound_basis_4_stride_2
    integer(i64), value :: bound_basis_4_stride_3
    integer(i64), value :: bound_basis_4_stride_4
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
    type(c_ptr), value :: bound_weights_3
    integer(i64), value :: bound_weights_3_shape_1
    integer(i64), value :: bound_weights_3_shape_2
    integer(i64), value :: bound_weights_3_stride_1
    integer(i64), value :: bound_weights_3_stride_2
    type(c_ptr), value :: bound_weights_4
    integer(i64), value :: bound_weights_4_shape_1
    integer(i64), value :: bound_weights_4_shape_2
    integer(i64), value :: bound_weights_4_stride_1
    integer(i64), value :: bound_weights_4_stride_2
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
    type(c_ptr), value :: bound_points_3
    integer(i64), value :: bound_points_3_shape_1
    integer(i64), value :: bound_points_3_shape_2
    integer(i64), value :: bound_points_3_stride_1
    integer(i64), value :: bound_points_3_stride_2
    type(c_ptr), value :: bound_points_4
    integer(i64), value :: bound_points_4_shape_1
    integer(i64), value :: bound_points_4_shape_2
    integer(i64), value :: bound_points_4_stride_1
    integer(i64), value :: bound_points_4_stride_2
    type(c_ptr), value :: bound_knots_1
    integer(i64), value :: bound_knots_1_shape_1
    integer(i64), value :: bound_knots_1_stride_1
    type(c_ptr), value :: bound_knots_2
    integer(i64), value :: bound_knots_2_shape_1
    integer(i64), value :: bound_knots_2_stride_1
    type(c_ptr), value :: bound_knots_3
    integer(i64), value :: bound_knots_3_shape_1
    integer(i64), value :: bound_knots_3_stride_1
    type(c_ptr), value :: bound_knots_4
    integer(i64), value :: bound_knots_4_shape_1
    integer(i64), value :: bound_knots_4_stride_1
    type(c_ptr), value :: bound_vector_d
    integer(i64), value :: bound_vector_d_shape_1
    integer(i64), value :: bound_vector_d_shape_2
    integer(i64), value :: bound_vector_d_stride_1
    integer(i64), value :: bound_vector_d_stride_2
    real(f64), value :: ovlp_value
    real(f64), value :: S_DDM
    integer(i64), value :: domain_nb
    type(c_ptr), value :: bound_rhs
    integer(i64), value :: bound_rhs_shape_1
    integer(i64), value :: bound_rhs_shape_2
    integer(i64), value :: bound_rhs_stride_1
    integer(i64), value :: bound_rhs_stride_2
    integer(i64), pointer :: spans_1(:)
    integer(i64), pointer :: spans_2(:)
    integer(i64), pointer :: spans_3(:)
    integer(i64), pointer :: spans_4(:)
    real(f64), pointer :: basis_1(:,:,:,:)
    real(f64), pointer :: basis_2(:,:,:,:)
    real(f64), pointer :: basis_3(:,:,:,:)
    real(f64), pointer :: basis_4(:,:,:,:)
    real(f64), pointer :: weights_1(:,:)
    real(f64), pointer :: weights_2(:,:)
    real(f64), pointer :: weights_3(:,:)
    real(f64), pointer :: weights_4(:,:)
    real(f64), pointer :: points_1(:,:)
    real(f64), pointer :: points_2(:,:)
    real(f64), pointer :: points_3(:,:)
    real(f64), pointer :: points_4(:,:)
    real(f64), pointer :: knots_1(:)
    real(f64), pointer :: knots_2(:)
    real(f64), pointer :: knots_3(:)
    real(f64), pointer :: knots_4(:)
    real(f64), pointer :: vector_d(:,:)
    real(f64), pointer :: rhs(:,:)

    call C_F_Pointer(bound_spans_1, spans_1, [bound_spans_1_shape_1 * &
          bound_spans_1_stride_1])
    call C_F_Pointer(bound_spans_2, spans_2, [bound_spans_2_shape_1 * &
          bound_spans_2_stride_1])
    call C_F_Pointer(bound_spans_3, spans_3, [bound_spans_3_shape_1 * &
          bound_spans_3_stride_1])
    call C_F_Pointer(bound_spans_4, spans_4, [bound_spans_4_shape_1 * &
          bound_spans_4_stride_1])
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
    call C_F_Pointer(bound_basis_3, basis_3, [bound_basis_3_shape_4 * &
          bound_basis_3_stride_4,bound_basis_3_shape_3 * &
          bound_basis_3_stride_3,bound_basis_3_shape_2 * &
          bound_basis_3_stride_2,bound_basis_3_shape_1 * &
          bound_basis_3_stride_1])
    call C_F_Pointer(bound_basis_4, basis_4, [bound_basis_4_shape_4 * &
          bound_basis_4_stride_4,bound_basis_4_shape_3 * &
          bound_basis_4_stride_3,bound_basis_4_shape_2 * &
          bound_basis_4_stride_2,bound_basis_4_shape_1 * &
          bound_basis_4_stride_1])
    call C_F_Pointer(bound_weights_1, weights_1, [ &
          bound_weights_1_shape_2 * bound_weights_1_stride_2, &
          bound_weights_1_shape_1 * bound_weights_1_stride_1])
    call C_F_Pointer(bound_weights_2, weights_2, [ &
          bound_weights_2_shape_2 * bound_weights_2_stride_2, &
          bound_weights_2_shape_1 * bound_weights_2_stride_1])
    call C_F_Pointer(bound_weights_3, weights_3, [ &
          bound_weights_3_shape_2 * bound_weights_3_stride_2, &
          bound_weights_3_shape_1 * bound_weights_3_stride_1])
    call C_F_Pointer(bound_weights_4, weights_4, [ &
          bound_weights_4_shape_2 * bound_weights_4_stride_2, &
          bound_weights_4_shape_1 * bound_weights_4_stride_1])
    call C_F_Pointer(bound_points_1, points_1, [bound_points_1_shape_2 * &
          bound_points_1_stride_2,bound_points_1_shape_1 * &
          bound_points_1_stride_1])
    call C_F_Pointer(bound_points_2, points_2, [bound_points_2_shape_2 * &
          bound_points_2_stride_2,bound_points_2_shape_1 * &
          bound_points_2_stride_1])
    call C_F_Pointer(bound_points_3, points_3, [bound_points_3_shape_2 * &
          bound_points_3_stride_2,bound_points_3_shape_1 * &
          bound_points_3_stride_1])
    call C_F_Pointer(bound_points_4, points_4, [bound_points_4_shape_2 * &
          bound_points_4_stride_2,bound_points_4_shape_1 * &
          bound_points_4_stride_1])
    call C_F_Pointer(bound_knots_1, knots_1, [bound_knots_1_shape_1 * &
          bound_knots_1_stride_1])
    call C_F_Pointer(bound_knots_2, knots_2, [bound_knots_2_shape_1 * &
          bound_knots_2_stride_1])
    call C_F_Pointer(bound_knots_3, knots_3, [bound_knots_3_shape_1 * &
          bound_knots_3_stride_1])
    call C_F_Pointer(bound_knots_4, knots_4, [bound_knots_4_shape_1 * &
          bound_knots_4_stride_1])
    call C_F_Pointer(bound_vector_d, vector_d, [bound_vector_d_shape_2 * &
          bound_vector_d_stride_2,bound_vector_d_shape_1 * &
          bound_vector_d_stride_1])
    call C_F_Pointer(bound_rhs, rhs, [bound_rhs_shape_2 * &
          bound_rhs_stride_2,bound_rhs_shape_1 * bound_rhs_stride_1])
    call assemble_vector_ex01(ne1 = ne1, ne2 = ne2, ne3 = ne3, ne4 = ne4 &
          , p1 = p1, p2 = p2, p3 = p3, p4 = p4, spans_1 = spans_1(1_i64 &
          ::bound_spans_1_stride_1), spans_2 = spans_2(1_i64:: &
          bound_spans_2_stride_1), spans_3 = spans_3(1_i64:: &
          bound_spans_3_stride_1), spans_4 = spans_4(1_i64:: &
          bound_spans_4_stride_1), basis_1 = basis_1(1_i64:: &
          bound_basis_1_stride_4, 1_i64::bound_basis_1_stride_3, 1_i64 &
          ::bound_basis_1_stride_2, 1_i64::bound_basis_1_stride_1), &
          basis_2 = basis_2(1_i64::bound_basis_2_stride_4, 1_i64:: &
          bound_basis_2_stride_3, 1_i64::bound_basis_2_stride_2, 1_i64 &
          ::bound_basis_2_stride_1), basis_3 = basis_3(1_i64:: &
          bound_basis_3_stride_4, 1_i64::bound_basis_3_stride_3, 1_i64 &
          ::bound_basis_3_stride_2, 1_i64::bound_basis_3_stride_1), &
          basis_4 = basis_4(1_i64::bound_basis_4_stride_4, 1_i64:: &
          bound_basis_4_stride_3, 1_i64::bound_basis_4_stride_2, 1_i64 &
          ::bound_basis_4_stride_1), weights_1 = weights_1(1_i64:: &
          bound_weights_1_stride_2, 1_i64::bound_weights_1_stride_1), &
          weights_2 = weights_2(1_i64::bound_weights_2_stride_2, 1_i64 &
          ::bound_weights_2_stride_1), weights_3 = weights_3(1_i64:: &
          bound_weights_3_stride_2, 1_i64::bound_weights_3_stride_1), &
          weights_4 = weights_4(1_i64::bound_weights_4_stride_2, 1_i64 &
          ::bound_weights_4_stride_1), points_1 = points_1(1_i64:: &
          bound_points_1_stride_2, 1_i64::bound_points_1_stride_1), &
          points_2 = points_2(1_i64::bound_points_2_stride_2, 1_i64:: &
          bound_points_2_stride_1), points_3 = points_3(1_i64:: &
          bound_points_3_stride_2, 1_i64::bound_points_3_stride_1), &
          points_4 = points_4(1_i64::bound_points_4_stride_2, 1_i64:: &
          bound_points_4_stride_1), knots_1 = knots_1(1_i64:: &
          bound_knots_1_stride_1), knots_2 = knots_2(1_i64:: &
          bound_knots_2_stride_1), knots_3 = knots_3(1_i64:: &
          bound_knots_3_stride_1), knots_4 = knots_4(1_i64:: &
          bound_knots_4_stride_1), vector_d = vector_d(1_i64:: &
          bound_vector_d_stride_2, 1_i64::bound_vector_d_stride_1), &
          ovlp_value = ovlp_value, S_DDM = S_DDM, domain_nb = domain_nb &
          , rhs = rhs(1_i64::bound_rhs_stride_2, 1_i64:: &
          bound_rhs_stride_1))

  end subroutine bind_c_assemble_vector_ex01
  !........................................

end module bind_c_mod_9edfdg6ic2th
