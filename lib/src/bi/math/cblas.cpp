/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "cblas.hpp"

#define CBLAS_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(cblas_##sname) *bi::cblas_##name<float>::func = cblas_##sname; \
  BOOST_TYPEOF(cblas_##dname) *bi::cblas_##name<double>::func = cblas_##dname;

CBLAS_FUNC_DEF(copy, dcopy, scopy)
CBLAS_FUNC_DEF(scal, dscal, sscal)
CBLAS_FUNC_DEF(dot, ddot, sdot)
CBLAS_FUNC_DEF(iamax, idamax, isamax)
CBLAS_FUNC_DEF(axpy, daxpy, saxpy)
CBLAS_FUNC_DEF(gemv, dgemv, sgemv)
CBLAS_FUNC_DEF(symv, dsymv, ssymv)
CBLAS_FUNC_DEF(trmv, dtrmv, strmv)
CBLAS_FUNC_DEF(gbmv, dgbmv, sgbmv)
CBLAS_FUNC_DEF(gemm, dgemm, sgemm)
CBLAS_FUNC_DEF(symm, dsymm, ssymm)
CBLAS_FUNC_DEF(trmm, dtrmm, strmm)
CBLAS_FUNC_DEF(ger, dger, sger)
CBLAS_FUNC_DEF(syr, dsyr, ssyr)
CBLAS_FUNC_DEF(syr2, dsyr2, ssyr2)
CBLAS_FUNC_DEF(syrk, dsyrk, ssyrk)
CBLAS_FUNC_DEF(trsv, dtrsv, strsv)
CBLAS_FUNC_DEF(trsm, dtrsm, strsm)
