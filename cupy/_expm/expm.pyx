
from cupy._core cimport _accelerator
from cupy._core._carray cimport shape_t
from cupy._core._dtype cimport to_cuda_dtype
from cupy._core._scalar cimport get_typename
from cupy._core.core cimport _internal_ascontiguousarray
from cupy._core.core cimport _ndarray_init
from cupy._core.core cimport ascontiguousarray
from cupy._core.core cimport ndarray
from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core cimport _routines_math as _math
from cupy.cuda cimport device
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.libs cimport cublas



cdef extern from '../../cupy_backends/cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y


cdef int _cuda_runtime_version = -1


cdef list compute_types = [COMPUTE_TYPE_TBD,  # float16
                           COMPUTE_TYPE_TBD,  # float32
                           COMPUTE_TYPE_TBD]  # float64
cdef dict compute_type_str = {
    0: 'COMPUTE_TYPE_TBD',
    1: 'COMPUTE_TYPE_DEFAULT',
    2: 'COMPUTE_TYPE_PEDANTIC',
    3: 'COMPUTE_TYPE_FP16',
    4: 'COMPUTE_TYPE_FP32',
    5: 'COMPUTE_TYPE_FP64',
    6: 'COMPUTE_TYPE_BF16',
    7: 'COMPUTE_TYPE_TF32',
}


cpdef int to_compute_type_index(dtype) except -1:
    cdef str dtype_char = numpy.dtype(dtype).char
    if dtype_char == 'e':
        return 0
    elif dtype_char in 'fF':
        return 1
    elif dtype_char in 'dD':
        return 2
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))


cpdef set_compute_type(dtype, compute_type):
    global compute_types
    if compute_type in (COMPUTE_TYPE_TBD, COMPUTE_TYPE_DEFAULT,
                        COMPUTE_TYPE_PEDANTIC, COMPUTE_TYPE_FP16,
                        COMPUTE_TYPE_FP32, COMPUTE_TYPE_FP64):
        compute_types[to_compute_type_index(dtype)] = compute_type
    elif compute_type in (COMPUTE_TYPE_BF16, COMPUTE_TYPE_TF32):
        if int(device.get_compute_capability()) >= 80:
            compute_types[to_compute_type_index(dtype)] = compute_type
        else:
            warnings.warn('COMPUTE_TYPE_BF16 and COMPUTE_TYPE_TF32 are only '
                          'available on GPUs with compute capability 8.0 or '
                          'higher. COMPUTE_TYPE_DEFAULT will be used instead.')
            compute_types[to_compute_type_index(dtype)] = COMPUTE_TYPE_DEFAULT
    else:
        raise ValueError('Unknown compute type: {}'.format(compute_type))


cpdef compute_type_to_str(compute_type):
    if compute_type in compute_type_str:
        return compute_type_str[compute_type]
    else:
        return compute_type

cdef ndarray _ell(CUDAMatrix& A, double coef, int m) {

    	ndarray sA = cp.eye(A);
    
    	double scale = np.pow(coef, (1 / (double) (2 * m + 1)));
        sA *= scale
    	#double alpha = sA.getNormAm(2 * m + 1) / A.getNorm(1);     #2 LINES BELOW ARE TEMPORARY REPLACEMENT
    	cp.pow(sA, (2 * m + 1), sA);
    	double alpha = sA.getNorm(1) / (double) (A.getNorm(1));
    	
        
         	return utils::max((int) (ceil(log2(2 * alpha / std::numeric_limits<double>::epsilon()) / (2 * m))), 0);
    }







cpdef get_pade_params(A):
    if out is not None:
        raise NotImplementedError('The out array as input is currently not '
                                  'supported')

    cdef Py_ssize_t i, n, m, ka, kb, a_sh, b_sh, c_sh
    cdef Py_ssize_t batchCount, a_part_outshape, b_part_outshape
    cdef int orig_a_ndim, orig_b_ndim, a_ndim, b_ndim, ndim
    cdef ndarray ap, bp, outp, out_view
    cdef bint use_broadcast

    orig_a_ndim = a._shape.size()
    orig_b_ndim = b._shape.size()
    if orig_a_ndim == 0 or orig_b_ndim == 0:
        raise ValueError('Scalar operands are not allowed, use \'*\' instead')

    ndim = max(orig_a_ndim, orig_b_ndim)
    if ndim <= 2:
        return dot(a, b, out)

    orig_a = a
    orig_b = b
    a_part_outshape = b_part_outshape = 0
    if orig_a_ndim == 1:
        a = _manipulation._reshape(a, (1, a.size))
    else:
        a = a.view()
        a_part_outshape = a._shape[orig_a_ndim - 2]
    if orig_b_ndim == 1:
        b = _manipulation._reshape(b, (b.size, 1))
        ldout = 1
    else:
        b = b.view()
        b_part_outshape = ldout = b._shape[orig_b_ndim - 1]

    # expand dims
    a_ndim = a._shape.size()
    b_ndim = b._shape.size()


    #Init
    double d4, d6, d8, d10, eta1, eta3, eta4, eta5;
    int ar, ac	
    ar,ac = A.shape
    ndarray theta
	ndarray coef
    #Init P;
    padeParams p
    p.pow.resize(11)
    p.scale = 0
    #Get coefficients and theta values
    coef = [
        (1 / 100800.0),
        (1 / 10059033600.0),
        (1 / 4487938430976000.0),
        (1 / 5914384781877411840000.0),
        (1 / 113250775606021113483283660800000000.0)
    ]
    theta = [
        1.495585217958292e-002,
        2.539398330063230e-001,
        9.504178996162932e-001,
        2.097847961257068e+000,
        5.371920351148152e+000
    ]

    # Get powers of A
    p.pow[2] = ndarray((ar, ac), dtype=dtype)
    p.pow[4] =  ndarray((ar, ac), dtype=dtype)
    p.pow[6] =  ndarray((ar, ac), dtype=dtype)
    p.pow[8] =  ndarray((ar, ac), dtype=dtype)
    p.pow[10] =  ndarray((ar, ac), dtype=dtype)


    p.pow[2] =  A @ A
    p.pow[4] =  p.pow[2] @ p.pow[2]
    p.pow[6] =  p.pow[2] @ p.pow[4]
    p.pow[8] =  p.pow[4] @ p.pow[4]
    p.pow[10] =  p.pow[4] @ p.pow[6]



    # NOT IDEAL .. PERFORM GETNORM ON DEVICE IF POSSIBLE. THIS MEANS SYNCING BETWEEN HOST AND DEVICE IS UNNECESSARY
    
    # Find mVal
    d4 = cp.pow(cp.norm(p.pow[4],1), (1.0 / 4))
    d6 = cp.pow(cp.norm(p.pow[6],1), (1.0 / 6))
    eta1 = cp.max(d4, d6)


    if ((eta1 <= theta[0]) && (ell(A, coef[0], 3) == 0)) {
        p.mVal = 3;
        return p;
    }
    if ((eta1 <= theta[1]) && (ell(A, coef[1], 5) == 0)) {
        p.mVal = 5;
        return p;
    }
    if (true) { //(A.isSmall()) {
        d8 = std::pow(p.pow[8]->getNorm(1), (1.0 / 8));
    } else {
        //d8 = pow(p.pow[4]->getNormAm(2), (1.0 / 8));
    }
    eta3 = utils::max(d6, d8);
    if ((eta3 <= theta[2]) && (ell(A, coef[2], 7) == 0)) {
        p.mVal = 7;
        return p;
    }
    if ((eta3 <= theta[3]) && (ell(A, coef[3], 9) == 0)) {
        p.mVal = 9;
        return p;
    }
    if (true) { //(A.isSmall()) {
        d10 = std::pow(p.pow[10]->getNorm(1), (1.0 / 10));
    } else {
        //d10 = std::pow(p.pow[2]->getNormAm(5), (1.0 / 10));
    }
    // Find scaling factor
    eta4 = utils::max(d8, d10);
    eta5 = utils::min(eta3, eta4);
    p.scale = utils::max((int) (ceil(log2(eta5 / theta[4]))), 0);
    CUDAMatrix sA(ar, ac);
    double multiplier = 1.0 / std::pow(2, p.scale);
    CUDAMatrix::mul(A, multiplier, sA);
    p.scale += ell(sA, coef[4], 13);
    if (std::isinf((double) p.scale)) {
        std::cout << "S = INF" << std::endl;
        int exp;																		// THIS CODE IS NOT ERROR CHECKED!!!!!
        double t = std::frexp(A.getNorm(1) / theta[4], &exp);
        p.scale = exp - (t == 0.5);
    }
    p.mVal = 13;
    return p;
    }






# This section is ported nearly verbatim from Eigen's implementation:
# https://eigen.tuxfamily.org/dox/unsupported/MatrixExponential_8h_source.html


def _matrix_exp_pade3(matrix2):
  """3rd-order Pade approximant for matrix exponential."""
  b = [120.0, 60.0, 12.0]
  b = [cp.ndarray(x, matrix.dtype) for x in b] #constants
  ident = cp.eye(
      matrix.shape[-2],
      batch_shape=matrix.shape[:-2],
      dtype=matrix.dtype)
  tmp = matrix_2 + b[1] * ident
  matrix_u = cp.matmul(matrix, tmp)
  matrix_v = b[2] * matrix_2 + b[0] * ident
  return matrix_u, matrix_v


def _matrix_exp_pade5(matrix,matrix_2,matrix_4):
  """5th-order Pade approximant for matrix exponential."""
  b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
  b = [cp.ndarray(x, matrix.dtype) for x in b] #constants
  ident = cp.eye(
      matrix.shape[-2],
      batch_shape=matrix.shape[:-2],
      dtype=matrix.dtype)

  tmp = matrix_4 + b[3] * matrix_2 + b[1] * ident
  matrix_u = cp.matmul(matrix, tmp)
  matrix_v = b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
  return matrix_u, matrix_v


def _matrix_exp_pade7(matrix,matrix_2,matrix_4, matrix_6):
  """7th-order Pade approximant for matrix exponential."""
  b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0]
  b = [cp.ndarray(x, matrix.dtype) for x in b] #constants
  ident = cp.eye(
      matrix.shape[-2],
      batch_shape=matrix.shape[:-2],
      dtype=matrix.dtype)
 )
  tmp = matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident
  matrix_u = math_ops.matmul(matrix, tmp)
  matrix_v = b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
  return matrix_u, matrix_v


def _matrix_exp_pade9(matrix,matrix_2, matrix_4, matrix_6,matrix_8):
  """9th-order Pade approximant for matrix exponential."""
  b = [
      17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
      2162160.0, 110880.0, 3960.0, 90.0
  ]
  b = [cp.ndarray(x, matrix.dtype) for x in b] #constants
  ident = cp.eye(
      matrix.shape[-2],
      batch_shape=matrix.shape[:-2],
      dtype=matrix.dtype)
  tmp = (
      matrix_8 + b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 +
      b[1] * ident)
  matrix_u = math_ops.matmul(matrix, tmp)
  matrix_v = (
      b[8] * matrix_8 + b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 +
      b[0] * ident)
  return matrix_u, matrix_v


def _matrix_exp_pade13(matrix,matix_2,matrix_4,matrix_6):
  """13th-order Pade approximant for matrix exponential."""
  b = [
      64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
      1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
      33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0
  ]
  b = [cp.ndarray(x, matrix.dtype) for x in b] #constants
  ident = cp.eye(
      matrix.shape[-2],
      batch_shape=matrix.shape[:-2],
      dtype=matrix.dtype)
  
  tmp_u = (
      math_ops.matmul(matrix_6, matrix_6 + b[11] * matrix_4 + b[9] * matrix_2) +
      b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident)
  matrix_u = math_ops.matmul(matrix, tmp_u)
  tmp_v = b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2
  matrix_v = (
      math_ops.matmul(matrix_6, tmp_v) + b[6] * matrix_6 + b[4] * matrix_4 +
      b[2] * matrix_2 + b[0] * ident)
  return matrix_u, matrix_v




@tf_export('linalg.expm')
@dispatch.add_dispatch_support
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
  
    matrix = cp.array(input, name='input')
    if matrix.shape[-2:] == [0, 0]:
      return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
      batch_shape = array_ops.shape(matrix)[:-2]

    # reshaping the batch makes the where statements work better
    matrix = cp.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    l1_norm = cp.max(                        
        cp.sum(
            math_ops.abs(matrix),                              
                        axis=matrix.size - 2),
        axis=-1)[..., cp.newaxis, cp.newaxis]

    const = lambda x: x.as(l1_norm.dtype) # constant

    def _nest_where(vals, cases):
      assert len(vals) == len(cases) - 1
      if len(vals) == 1:
        return cp.where(
            l1_norm < const(vals[0]), cases[0], cases[1])
      else:
        return cp.where(
            l1_norm < const(vals[0]), cases[0],
            _nest_where(vals[1:], cases[1:]))

    p = {}
    p['matrix'] =matrix
    p['matrix_2'] =cp.matmul(matrix,matrix)
    p['matrix_4'] =cp.matmul(p['matrix_2'],p['matrix_2'])
    p['matrix_6'] =cp.matmul(p['matrix_4'],p['matrix_2'])
    p['matrix_8'] =cp.matmul(p['matrix_4'],p['matrix_4'])

    if matrix.dtype in [dtypes.float16, dtypes.float32, dtypes.complex64]:
      
      maxnorm = const(3.925724783138660)
      squarings = cp.max(
            cp.floor(
              math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
      u3, v3 = _matrix_exp_pade3(matrix,p['matrix_2'])
      u5, v5 = _matrix_exp_pade5(matrix,p['matrix_2'],p['matrix_4'])
      
      scale = 1/ cp.cast(cp.pow(const(2.0), squarings), matrix.dtype)
      u7, v7 = _matrix_exp_pade7(
          matrix *scale,,p['matrix_2'] *(scale**2),p['matrix_4']*(scale**4),p['matrix_6']*(scale**6))
      
      conds = (4.258730016922831e-001, 1.880152677804762e+000)
      u = _nest_where(conds, (u3, u5, u7))
      v = _nest_where(conds, (v3, v5, v7))
    
    elif matrix.dtype in [dtypes.float64, dtypes.complex128]:
    
      maxnorm = const(5.371920351148152)
      squarings = cp.max(
            cp.floor(
              cp.log(l1_norm / maxnorm) / cp.log(const(2.0))), 0)

      u3, v3 = _matrix_exp_pade3(matrix,p['matrix_2'])
      u5, v5 = _matrix_exp_pade5(matrix,p['matrix_2'],p['matrix_4'])
      u7, v7 = _matrix_exp_pade7(matrix,p['matrix_2'],p['matrix_4'],p['matrix_6'])
      u9, v9 = _matrix_exp_pade9(matrix,p['matrix_2'],p['matrix_4'],p['matrix_6'],p['matrix_8')
      
      scale = 1/ cp.cast(cp.pow(const(2.0), squarings), matrix.dtype)
      u13, v13 = _matrix_exp_pade13( matrix * scale,p['matrix_2'] *(scale**2),p['matrix_4']*(scale**4),p['matrix_6']*(scale**6))

      conds = (1.495585217958292e-002, 2.539398330063230e-001,
               9.504178996162932e-001, 2.097847961257068e+000)
      u = _nest_where(conds, (u3, u5, u7, u9, u13))
      v = _nest_where(conds, (v3, v5, v7, v9, v13))
    
    else:
      raise ValueError('tf.linalg.expm does not support matrices of type %s' %
                       matrix.dtype)

    is_finite = math_ops.is_finite(math_ops.reduce_max(l1_norm))
    nan = constant_op.constant(np.nan, matrix.dtype)
    result = control_flow_ops.cond(
        is_finite, lambda: cp.linalg.solve(-u + v, u + v),
        lambda: cp.fill(matrix), nan))
    max_squarings = cp.max(squarings)
    i = const(0.0)

    def c(i, _):
      return cp.where(is_finite,
                                   lambda: i < max_squarings,
                                   lambda: const(False))
    def b(i, r):
      return i + 1, cp.where(
          i < squarings, cp.matmul(r, r), r)

    _, result = where_loop(c, b, [i, result])

    if None in matrix.shape:
      return cp.reshape(
          result,
          cp.concat((batch_shape, result.shape[-2:]), axis=0))
    return result.reshape( batch_shape.concatenate(result.shape[-2:])))

def where_loop(condition, body, variables):

    while condtion(*variables):

        variables = body(variables)
    return variables





