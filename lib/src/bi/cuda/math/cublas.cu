/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "cublas.hpp"

#define CUBLAS_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(cublas##sname) *bi::cublas_##name<float>::func = cublas##sname; \
  BOOST_TYPEOF(cublas##dname) *bi::cublas_##name<double>::func = cublas##dname;

CUBLAS_FUNC_DEF(copy, Dcopy, Scopy)
CUBLAS_FUNC_DEF(scal, Dscal, Sscal)
CUBLAS_FUNC_DEF(dot, Ddot, Sdot)
CUBLAS_FUNC_DEF(iamax, Idamax, Isamax)
CUBLAS_FUNC_DEF(axpy, Daxpy, Saxpy)
CUBLAS_FUNC_DEF(gemv, Dgemv, Sgemv)
CUBLAS_FUNC_DEF(symv, Dsymv, Ssymv)
CUBLAS_FUNC_DEF(trmv, Dtrmv, Strmv)
CUBLAS_FUNC_DEF(gbmv, Dgbmv, Sgbmv)
CUBLAS_FUNC_DEF(gemm, Dgemm, Sgemm)
CUBLAS_FUNC_DEF(symm, Dsymm, Ssymm)
CUBLAS_FUNC_DEF(trmm, Dtrmm, Strmm)
CUBLAS_FUNC_DEF(ger, Dger, Sger)
CUBLAS_FUNC_DEF(syr, Dsyr, Ssyr)
CUBLAS_FUNC_DEF(syr2, Dsyr2, Ssyr2)
CUBLAS_FUNC_DEF(syrk, Dsyrk, Ssyrk)
CUBLAS_FUNC_DEF(trsv, Dtrsv, Strsv)
CUBLAS_FUNC_DEF(trsm, Dtrsm, Strsm)
