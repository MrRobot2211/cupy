import cupy
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import _util


# This section is ported nearly verbatim from Eigen's implementation:
# https://eigen.tuxfamily.org/dox/unsupported/MatrixExponential_8h_source.html


def _matrix_exp_pade3(matrix, matrix_2):
    """3rd-order Pade approximant for matrix exponential."""
    b = [120.0, 60.0, 12.0]
    b = [cupy.ndarray(x, matrix.dtype) for x in b]  # constants
    ident = cupy.eye(
        matrix.shape[-2],
        batch_shape=matrix.shape[:-2],
        dtype=matrix.dtype)
    tmp = matrix_2 + b[1] * ident
    matrix_u = cupy.matmul(matrix, tmp)
    matrix_v = b[2] * matrix_2 + b[0] * ident
    return matrix_u, matrix_v


def _matrix_exp_pade5(matrix, matrix_2, matrix_4):
    """5th-order Pade approximant for matrix exponential."""
    b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
    b = [cupy.ndarray(x, matrix.dtype) for x in b]  # constants
    ident = cupy.eye(
        matrix.shape[-2],
        batch_shape=matrix.shape[:-2],
        dtype=matrix.dtype)

    tmp = matrix_4 + b[3] * matrix_2 + b[1] * ident
    matrix_u = cupy.matmul(matrix, tmp)
    matrix_v = b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
    return matrix_u, matrix_v


def _matrix_exp_pade7(matrix, matrix_2, matrix_4, matrix_6):
    """7th-order Pade approximant for matrix exponential."""
    b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0]
    b = [cupy.ndarray(x, matrix.dtype) for x in b]  # constants
    ident = cupy.eye(
        matrix.shape[-2],
        batch_shape=matrix.shape[:-2],
        dtype=matrix.dtype)

    tmp = matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident
    matrix_u = matrix @ tmp
    matrix_v = b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
    return matrix_u, matrix_v


def _matrix_exp_pade9(matrix, matrix_2, matrix_4, matrix_6, matrix_8):
    """9th-order Pade approximant for matrix exponential."""
    b = [
        17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0, 110880.0, 3960.0, 90.0
    ]
    b = [cupy.ndarray(x, matrix.dtype) for x in b]  # constants
    ident = cupy.eye(
        matrix.shape[-2],
        batch_shape=matrix.shape[:-2],
        dtype=matrix.dtype)
    tmp = (
        matrix_8 + b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 +
        b[1] * ident)
    matrix_u = matrix @ tmp
    matrix_v = (
        b[8] * matrix_8 + b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 +
        b[0] * ident)
    return matrix_u, matrix_v


def _matrix_exp_pade13(matrix, matrix_2, matrix_4, matrix_6):
    """13th-order Pade approximant for matrix exponential."""
    b = [
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
        33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0
    ]
    b = [cupy.ndarray(x, matrix.dtype) for x in b]  # constants
    ident = cupy.eye(
        matrix.shape[-2],
        batch_shape=matrix.shape[:-2],
        dtype=matrix.dtype)

    tmp_u = (
        matrix_6 @ (matrix_6 + b[11] * matrix_4 + b[9] * matrix_2) +
        b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident)
    matrix_u = matrix @ tmp_u
    tmp_v = b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2
    matrix_v = (
        (matrix_6 @ tmp_v) + b[6] * matrix_6 + b[4] * matrix_4 +
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






def expm(a):
    """
    This method calculates the exponential of a square matrix, `` /exp{a}``.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.

    Returns:
        scalar :class:`~cupy.ndarray`:

    .. seealso:: :func:`scipy.linalg.expm`
    """

    a = cupy.asarray(a)
    _util._assert_2d(a)
    _util._assert_stacked_square(a)

    dtype = a.dtype

    if dtype.char == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
    elif dtype.char == 'd':
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize
    elif dtype.char == 'F':
        getrf = cusolver.cgetrf
        getrf_bufferSize = cusolver.cgetrf_bufferSize
    elif dtype.char == 'D':
        getrf = cusolver.zgetrf
        getrf_bufferSize = cusolver.zgetrf_bufferSize
    else:
        msg = 'Only float32, float64, complex64
    and complex128 are supported.'
        raise NotImplementedError(msg)

    a = a.astype(dtype, order='F', copy=(not overwrite_a))

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    m, n = a.shape

    ipiv = cupy.empty((min(m, n),), dtype=numpy.intc)

    buffersize = getrf_bufferSize(cusolver_handle, m, n, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)

def get_pade_params(A):


    ar,ac = A.shape


    # Get coefficients and theta values
    coef = {
        (1 / 100800.0),
        (1 / 10059033600.0),
        (1 / 4487938430976000.0),
        (1 / 5914384781877411840000.0),
        (1 / 113250775606021113483283660800000000.0)
    }
    theta = {
        1.495585217958292e-002,
        2.539398330063230e-001,
        9.504178996162932e-001,
        2.097847961257068e+000,
        5.371920351148152e+000
    }




get_pade_parameters = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void pade_parameter(complex<float>* A) { #CUDAMatrix& A

    // Init
    	double d4, d6, d8, d10, eta1, eta3, eta4, eta5;
    	int ar = A.getNumRows();
    	int ac = A.getNumCols();
    	std::vector<double> theta;
    	std::vector<double> coef;
    // Init P;
    padeParams p;
    p.pow.resize(11);
    p.scale = 0;
    // Get coefficients and theta values
    coef = {
        (1 / 100800.0),
        (1 / 10059033600.0),
        (1 / 4487938430976000.0),
        (1 / 5914384781877411840000.0),
        (1 / 113250775606021113483283660800000000.0)
    };
    theta = {
        1.495585217958292e-002,
        2.539398330063230e-001,
        9.504178996162932e-001,
        2.097847961257068e+000,
        5.371920351148152e+000
    };
    // Get powers of A
    p.pow[2] = new CUDAMatrix(ar, ac);
    p.pow[4] = new CUDAMatrix(ar, ac);
    p.pow[6] = new CUDAMatrix(ar, ac);
    p.pow[8] = new CUDAMatrix(ar, ac);
    p.pow[10] = new CUDAMatrix(ar, ac);
    cudaParams cp = getCUDAParams(A.getNumRows(), A.getNumCols());
    cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, A.d_matrix, p.pow[2]->d_matrix, ar);
    cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[2]->d_matrix, p.pow[2]->d_matrix, p.pow[4]->d_matrix, ar);
    cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[2]->d_matrix, p.pow[4]->d_matrix, p.pow[6]->d_matrix, ar);
    cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[4]->d_matrix, p.pow[4]->d_matrix, p.pow[8]->d_matrix, ar);
    cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[4]->d_matrix, p.pow[6]->d_matrix, p.pow[10]->d_matrix, ar);

    // NOT IDEAL .. PERFORM GETNORM ON DEVICE IF POSSIBLE. THIS MEANS SYNCING BETWEEN HOST AND DEVICE IS UNNECESSARY
    p.pow[2]->syncHost();
    p.pow[4]->syncHost();
    p.pow[6]->syncHost();
    p.pow[8]->syncHost();
    p.pow[10]->syncHost();
    ////

    // Find mVal
    d4 = std::pow(p.pow[4]->getNorm(1), (1.0 / 4));
    d6 = std::pow(p.pow[6]->getNorm(1), (1.0 / 6));
    eta1 = utils::max(d4, d6);
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
    ''', 'pade_parameter')


    #     CUDATimer CUDAMatrix::exp(CUDAMatrix& A, CUDAMatrix& R) {
    # 	if (A.isInitialised() && R.isInitialised()) {
    # 		int ar = A.getNumRows();
    # 		int ac = A.getNumCols();
    # 		int rr = R.getNumRows();
    # 		int rc = R.getNumCols();
    # 		if (ar == ac && ac == rr && rr == rc) {
    # 			A.syncDevice();
    # 			CUDATimer t;
    # 			int c1, c2;
    # 			int n = utils::max(ar, ac);
    # 			// Special Cases
    # 			if (A.isDiagonal()) {
    # 				t.start();
    # 				for (c1 = 0; c1 < n; c1++) {
    # 					R.setCell(c1, c1, std::exp(A.getCell(c1, c1)));
    # 				}
    # 				t.stop();
    # 				R.syncDevice();
    # 			} else if (A.isZero()) {
    # 				t.start();
    # 				R.setMatrix(0);
    # 				t.stop();
    # 				R.syncDevice();
    # 			// Normal Case
    # 			} else {
    # 				// Create Matrices
    # 				CUDAMatrix U(ar, ac);
    # 				CUDAMatrix V(ar, ac);
    # 				CUDAMatrix I(ar, ac); // Identity
    # 				CUDAMatrix T(ar, ac); // Tally
    # 				CUDAMatrix TMP(ar, ac); // Temporary
    # 				I.setIdentity();
    # 				I.syncDevice();
    # 				// Get CUDA params
    # 				cudaParams cp = getCUDAParams(ar, ac);
    # 				// Get Pade params
    # 				padeParams p = getPadeParams(A);
    # 				int s = p.scale;
    # 				int m = p.mVal;
    # 				std::vector<CUDAMatrix*> pow = p.pow;
    # 				// Get Pade coefficients
    # 				std::vector<double> c = getPadeCoefficients(m);
    # 				// Start timer
    # 				t.start();
    # 				// Scaling
    # 				if (s != 0) {
    # 					double multiplier;
    # 					multiplier = 1.0 / std::pow(2, s);
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, multiplier, A.d_matrix, n);
    # 					for (c1 = 2; c1 <= 6; c1 += 2) {
    # 						multiplier = 1.0 / std::pow(2, (s * c1));
    # 						cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c1]->d_matrix, multiplier, pow[c1]->d_matrix, n);
    # 					}
    # 				}
    # 				// Approximation
    # 				if (m == 3 || m == 5 || m == 7 || m == 9) {
    # 					for (c1 = (int) (pow.size()) + 2; c1 < m - 1; c1 += 2) { //for (k = strt:2:m-1)
    # 						cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c1 - 2]->d_matrix, pow[2]->d_matrix, pow[c1]->d_matrix, n);
    # 					}
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[1], U.d_matrix, n);
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[0], V.d_matrix, n);
    # 					for (c2 = m; c2 >= 3; c2 -= 2) { //for (j = m : -2 : 3)
    # 						cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c2 - 1]->d_matrix, c[c2], TMP.d_matrix, n);
    # 						cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (U.d_matrix, TMP.d_matrix, U.d_matrix, n);
    # 						cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[c2 - 1]->d_matrix, c[c2-1], TMP.d_matrix, n);
    # 						cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (V.d_matrix, TMP.d_matrix, V.d_matrix, n);
    # 					}
    # 					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (U.d_matrix, A.d_matrix, U.d_matrix, n);
    # 				} else if (m == 13) {
    # 					// This is the equivellent of ..
    # 					// U = A * (p[6] * (c[13] * p[6] + c[11] * p[4] + c[9] * p[2]) + c[7] * p[6] + c[5] * p[4] + c[3] * p[2] + c[1] * I);		RUN IN STREAM 1
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[13], T.d_matrix, n);		// p[6] * c[13] -> T			Needs new TMP var
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[11], TMP.d_matrix, n);		// p[4] * c[11] -> TMP			(Cannot be used in multiple streams)
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[9], TMP.d_matrix, n);		// p[2] * c[9]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, T.d_matrix, T.d_matrix, n);			// p[6] * T     -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[7], TMP.d_matrix, n);		// p[6] * c[7]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[5], TMP.d_matrix, n);		// p[4] * c[5]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[3], TMP.d_matrix, n);		// p[2] * c[3]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[1], TMP.d_matrix, n);				// I * c[1]     -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, T.d_matrix, U.d_matrix, n);				// A * T        -> U
    # 					// This is the equivellent of ..
    # 					//V = p[6] * (c[12] * p[6] + c[10] * p[4] + c[8] * p[2]) + c[6] * p[6] + c[4] * p[4] + c[2] * p[2] + c[0] * I;				RUN IN STREAM 2
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[12], T.d_matrix, n);		// p[6] * c[12] -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[10], TMP.d_matrix, n);		// p[4] * c[10] -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[8], TMP.d_matrix, n);		// p[2] * c[8]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, T.d_matrix, T.d_matrix, n);			// p[6]			-> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[6]->d_matrix, c[6], TMP.d_matrix, n);		// p[6] * c[6]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[4]->d_matrix, c[4], TMP.d_matrix, n);		// p[4] * c[4]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (pow[2]->d_matrix, c[2], TMP.d_matrix, n);		// p[2] * c[2]  -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);				// T + TMP      -> T
    # 					cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (I.d_matrix, c[0], TMP.d_matrix, n);				// I * c[0]     -> TMP
    # 					cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, V.d_matrix, n);				// T + TMP      -> V
    # 				}
    # 				// This is the equivellent of ..
    # 				// R = (V - U) / (2 * U) + I;  ||?? R = (-U + V) / (U + V);
    # 				cudaSub KERNEL_ARGS2(cp.bpg, cp.tpb) (V.d_matrix, U.d_matrix, T.d_matrix, n);
    # 				cudaMulScalar KERNEL_ARGS2(cp.bpg, cp.tpb) (U.d_matrix, 2, TMP.d_matrix, n);
    # 				//cudaInv KERNEL_ARGS2(cp.bpg, cp.tpb) (TMP.d_matrix, TMP.d_matrix, n); // TEMP CODE BELOW
    # 				T.syncHost();
    # 				CUDAMatrix::inv(T, T);
    # 				T.syncDevice();
    # 				//
    # 				cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, TMP.d_matrix, T.d_matrix, n);
    # 				cudaAdd KERNEL_ARGS2(cp.bpg, cp.tpb) (T.d_matrix, I.d_matrix, R.d_matrix, n);
    # 				// Squaring
    # 				for (int k = 0; k < s; k++) {
    # 					cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (R.d_matrix, R.d_matrix, R.d_matrix, n);
    # 				}
    # 				cudaThreadSynchronize();
    # 				t.stop();
    # 				R.syncHost();
    # 			}
    # 			return t;
    # 		} else {
    # 			throw std::runtime_error("Matrix sizez do not match");
    # 		}
    # 	} else {
    # 		throw std::runtime_error("Cannot perform matrix operations before initialisation");
    # 	}
    # }

    # // INTERNAL PADE APPROXIMATION CODE

    # int CUDAMatrix::ell(CUDAMatrix& A, double coef, int m) {
    # 	CUDAMatrix sA(A.getNumRows());
    # 	CUDAMatrix::abs(A, sA);
    # 	double scale = std::pow(coef, (1 / (double) (2 * m + 1)));
    # 	CUDAMatrix::mul(sA, scale, sA);
    # 	//double alpha = sA.getNormAm(2 * m + 1) / A.getNorm(1);     2 LINES BELOW ARE TEMPORARY REPLACEMENT
    # 	CUDAMatrix::pow(sA, (2 * m + 1), sA);
    # 	double alpha = sA.getNorm(1) / (double) (A.getNorm(1));
    # 	/////
    # 	return utils::max((int) (ceil(log2(2 * alpha / std::numeric_limits<double>::epsilon()) / (2 * m))), 0);
    # }

    # CUDAMatrix::padeParams CUDAMatrix::getPadeParams(CUDAMatrix& A) {
    # 	// Init
    # 	double d4, d6, d8, d10, eta1, eta3, eta4, eta5;
    # 	int ar = A.getNumRows();
    # 	int ac = A.getNumCols();
    # 	std::vector<double> theta;
    # 	std::vector<double> coef;
    # 	// Init P;
    # 	padeParams p;
    # 	p.pow.resize(11);
    # 	p.scale = 0;
    # 	// Get coefficients and theta values
    # 	coef = {
    # 		(1 / 100800.0),
    # 		(1 / 10059033600.0),
    # 		(1 / 4487938430976000.0),
    # 		(1 / 5914384781877411840000.0),
    # 		(1 / 113250775606021113483283660800000000.0)
    # 	};
    # 	theta = {
    # 		1.495585217958292e-002,
    # 		2.539398330063230e-001,
    # 		9.504178996162932e-001,
    # 		2.097847961257068e+000,
    # 		5.371920351148152e+000
    # 	};
    # 	// Get powers of A
    # 	p.pow[2] = new CUDAMatrix(ar, ac);
    # 	p.pow[4] = new CUDAMatrix(ar, ac);
    # 	p.pow[6] = new CUDAMatrix(ar, ac);
    # 	p.pow[8] = new CUDAMatrix(ar, ac);
    # 	p.pow[10] = new CUDAMatrix(ar, ac);
    # 	cudaParams cp = getCUDAParams(A.getNumRows(), A.getNumCols());
    # 	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (A.d_matrix, A.d_matrix, p.pow[2]->d_matrix, ar);
    # 	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[2]->d_matrix, p.pow[2]->d_matrix, p.pow[4]->d_matrix, ar);
    # 	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[2]->d_matrix, p.pow[4]->d_matrix, p.pow[6]->d_matrix, ar);
    # 	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[4]->d_matrix, p.pow[4]->d_matrix, p.pow[8]->d_matrix, ar);
    # 	cudaMul KERNEL_ARGS2(cp.bpg, cp.tpb) (p.pow[4]->d_matrix, p.pow[6]->d_matrix, p.pow[10]->d_matrix, ar);

    # 	// NOT IDEAL .. PERFORM GETNORM ON DEVICE IF POSSIBLE. THIS MEANS SYNCING BETWEEN HOST AND DEVICE IS UNNECESSARY
    # 	p.pow[2]->syncHost();
    # 	p.pow[4]->syncHost();
    # 	p.pow[6]->syncHost();
    # 	p.pow[8]->syncHost();
    # 	p.pow[10]->syncHost();
    # 	////

    # 	// Find mVal
    # 	d4 = std::pow(p.pow[4]->getNorm(1), (1.0 / 4));
    # 	d6 = std::pow(p.pow[6]->getNorm(1), (1.0 / 6));
    # 	eta1 = utils::max(d4, d6);
    # 	if ((eta1 <= theta[0]) && (ell(A, coef[0], 3) == 0)) {
    # 		p.mVal = 3;
    # 		return p;
    # 	}
    # 	if ((eta1 <= theta[1]) && (ell(A, coef[1], 5) == 0)) {
    # 		p.mVal = 5;
    # 		return p;
    # 	}
    # 	if (true) { //(A.isSmall()) {
    # 		d8 = std::pow(p.pow[8]->getNorm(1), (1.0 / 8));
    # 	} else {
    # 		//d8 = pow(p.pow[4]->getNormAm(2), (1.0 / 8));
    # 	}
    # 	eta3 = utils::max(d6, d8);
    # 	if ((eta3 <= theta[2]) && (ell(A, coef[2], 7) == 0)) {
    # 		p.mVal = 7;
    # 		return p;
    # 	}
    # 	if ((eta3 <= theta[3]) && (ell(A, coef[3], 9) == 0)) {
    # 		p.mVal = 9;
    # 		return p;
    # 	}
    # 	if (true) { //(A.isSmall()) {
    # 		d10 = std::pow(p.pow[10]->getNorm(1), (1.0 / 10));
    # 	} else {
    # 		//d10 = std::pow(p.pow[2]->getNormAm(5), (1.0 / 10));
    # 	}
    # 	// Find scaling factor
    # 	eta4 = utils::max(d8, d10);
    # 	eta5 = utils::min(eta3, eta4);
    # 	p.scale = utils::max((int) (ceil(log2(eta5 / theta[4]))), 0);
    # 	CUDAMatrix sA(ar, ac);
    # 	double multiplier = 1.0 / std::pow(2, p.scale);
    # 	CUDAMatrix::mul(A, multiplier, sA);
    # 	p.scale += ell(sA, coef[4], 13);
    # 	if (std::isinf((double) p.scale)) {
    # 		std::cout << "S = INF" << std::endl;
    # 		int exp;																		// THIS CODE IS NOT ERROR CHECKED!!!!!
    # 		double t = std::frexp(A.getNorm(1) / theta[4], &exp);
    # 		p.scale = exp - (t == 0.5);
    # 	}
    # 	p.mVal = 13;
    # 	return p;
    # }

    # std::vector<double> CUDAMatrix::getPadeCoefficients(int m) {
    # 	switch (m) {
    # 		case 3:
    # 			return { 120, 60, 12, 1 };
    # 		case 5:
    # 			return { 30240, 15120, 3360, 420, 30, 1 };
    # 		case 7:
    # 			return { 17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1 };
    # 		case 9:
    # 			return { 17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1 };
    # 		case 13:
    # 			return { 64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1 };
    # 		default:
    # 			throw std::runtime_error("Invalid m value");
    # 	}
    # }

    pass
