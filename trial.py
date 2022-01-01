

def matrix_exponential(input, name=None):  # pylint: disable=redefined-builtin
    r"""Computes the matrix exponential of one or more square matrices.
    $$exp(A) = \sum_{n=0}^\infty A^n/n!$$
    The exponential is computed using a combination of the scaling and squaring
    method and the Pade approximation. Details can be found in:
    Nicholas J. Higham, "The scaling and squaring method for the matrix
    exponential revisited," SIAM J. Matrix Anal. Applic., 26:1179-1193, 2005.
    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices. The output is a tensor of the same shape as the input
    containing the exponential for all input submatrices `[..., :, :]`.
    Args:
        input: A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`, or
        `complex128` with shape `[..., M, M]`.
        name:  A name to give this `Op` (optional).
    Returns:
        the matrix exponential of the input.
    Raises:
        ValueError: An unsupported type is provided as input.
    @compatibility(scipy)
    Equivalent to scipy.linalg.expm
    @end_compatibility
    """

    matrix = cupy.array(input, name='input')
    if matrix.shape[-2:] == [0, 0]:
        return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
        batch_shape = array_ops.shape(matrix)[:-2]

    # reshaping the batch makes the where statements work better
    matrix = cupy.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    l1_norm = cupy.max(                        
        cupy.sum(
            math_ops.abs(matrix),                              
                        axis=matrix.size - 2),
        axis=-1)[..., cupy.newaxis, cupy.newaxis]

    const = lambda x: x.as(l1_norm.dtype) # constant

    def _nest_where(vals, cases):
      assert len(vals) == len(cases) - 1
      if len(vals) == 1:
        return cupy.where(
            l1_norm < const(vals[0]), cases[0], cases[1])
      else:
        return cupy.where(
            l1_norm < const(vals[0]), cases[0],
            _nest_where(vals[1:], cases[1:]))

    p = {}
    p['matrix'] = matrix
    p['matrix_2'] = cupy.matmul(matrix, matrix)
    p['matrix_4'] = cupy.matmul(p['matrix_2'], p['matrix_2'])
    p['matrix_6'] = cupy.matmul(p['matrix_4'], p['matrix_2'])
    p['matrix_8'] = cupy.matmul(p['matrix_4'], p['matrix_4'])

    if matrix.dtype in [cupy.float16, cupy.float32, cupy.complex64]:

        maxnorm = const(3.925724783138660)
        squarings = cupy.max(
                cupy.floor(
                cupy.log(l1_norm / maxnorm) / cupy.log(const(2.0))), 0)
        u3, v3 = _matrix_exp_pade3(matrix, p['matrix_2'])
        u5, v5 = _matrix_exp_pade5(matrix, p['matrix_2'], p['matrix_4'])

        scale = 1/ cupy.cast(cupy.pow(const(2.0), squarings), matrix.dtype)
        u7, v7 = _matrix_exp_pade7(
            matrix *scale, p['matrix_2'] *(scale**2), p['matrix_4']*(scale**4), p['matrix_6']*(scale**6))

        conds = (4.258730016922831e-001, 1.880152677804762e+000)
        u = _nest_where(conds, (u3, u5, u7))
        v = _nest_where(conds, (v3, v5, v7))

    elif matrix.dtype in [cupy.float64, cupy.complex128]:

        maxnorm = const(5.371920351148152)
        squarings = cupy.max(
                cupy.floor(
                cupy.log(l1_norm / maxnorm) / cupy.log(const(2.0))), 0)

        u3, v3 = _matrix_exp_pade3(matrix, p['matrix_2'])
        u5, v5 = _matrix_exp_pade5(matrix, p['matrix_2'], p['matrix_4'])
        u7, v7 = _matrix_exp_pade7(matrix, p['matrix_2'], p['matrix_4'], p['matrix_6'])
        u9, v9 = _matrix_exp_pade9(matrix, p['matrix_2'], p['matrix_4'], p['matrix_6'], p['matrix_8'])

        scale = 1/ cupy.cast(cupy.pow(const(2.0), squarings), matrix.dtype)
        u13, v13 = _matrix_exp_pade13(matrix * scale, p['matrix_2'] *(scale**2), p['matrix_4']*(scale**4), p['matrix_6']*(scale**6))

        conds = (1.495585217958292e-002, 2.539398330063230e-001,
                9.504178996162932e-001, 2.097847961257068e+000)
        u = _nest_where(conds, (u3, u5, u7, u9, u13))
        v = _nest_where(conds, (v3, v5, v7, v9, v13))

    else:
        raise ValueError('tf.linalg.expm does not support matrices of type %s' %
                       matrix.dtype)

    is_finite = math_ops.is_finite(math_ops.reduce_max(l1_norm))
    nan = constant_op.constant(np.nan, matrix.dtype)
    result = cupy.where(
        is_finite, lambda: cupy.linalg.solve(-u + v, u + v),
        lambda: matrix.shape.fill(cupy.nan))
    max_squarings = cupy.max(squarings)
    i = const(0.0)

    def c(i, _):
      return cupy.where(is_finite,
                                   lambda: i < max_squarings,
                                   lambda: const(False))
    def b(i, r):
      return i + 1, cupy.where(
          i < squarings, cupy.matmul(r, r), r)

    _, result = where_loop(c, b, [i, result])

    if None in matrix.shape:
      return cupy.reshape(
          result,
          cupy.concat((batch_shape, result.shape[-2:]), axis=0))
    return result.reshape( batch_shape.concatenate(result.shape[-2:]))

def where_loop(condition, body, variables):

    while condtion(*variables):

        variables = body(variables)
    return variables