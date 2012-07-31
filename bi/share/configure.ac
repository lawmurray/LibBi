AC_INIT([bi], [1.0.0], [bug-report@example.com], [bi], [http://www.example.com])
AC_PREREQ([2.59])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Save user's flags
USER_CPPFLAGS=$CPPFLAGS
USER_CXXFLAGS=$CXXFLAGS
USER_LDFLAGS=$LDFLAGS

# Conditionals used by make
AC_ARG_ENABLE([single],
     [  --enable-single         use single-precision floating point],
     [case "${enableval}" in
       yes) single=true ;;
       no)  single=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-single]) ;;
     esac],[single=false])

AC_ARG_ENABLE([cuda],
     [  --enable-cuda           use CUDA code for compatible GPU device],
     [case "${enableval}" in
       yes) cuda=true ;;
       no)  cuda=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-cuda]) ;;
     esac],[cuda=false])

AC_ARG_ENABLE([sse],
     [  --enable-sse            use SSE code],
     [case "${enableval}" in
       yes) sse=true ;;
       no)  sse=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-sse]) ;;
     esac],[sse=false])

AC_ARG_ENABLE([mkl],
     [  --enable-mkl            use Intel MKL],
     [case "${enableval}" in
       yes) mkl=true ;;
       no)  mkl=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-mkl]) ;;
     esac],[mkl=false])
AM_CONDITIONAL([ENABLE_MKL], [test x$mkl = xtrue])

AC_ARG_ENABLE([mpi],
     [  --enable-mpi            use MPI code],
     [case "${enableval}" in
       yes) mpi=true ;;
       no)  mpi=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-mpi]) ;;
     esac],[mpi=false])

AC_ARG_ENABLE([vampir],
     [  --enable-vampir         use Vampir],
     [case "${enableval}" in
       yes) vampir=true ;;
       no)  vampir=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-vampir]) ;;
     esac],[vampir=false])
     
# Add standard CUDA directories
#if test x$cuda = xtrue; then
# ^ don't test, may be using Thrust bundled with CUDA, even for host
    CUDA_CPPFLAGS="-I/usr/local/cuda/include -I$CUDA_ROOT/include"
    CUDA_CXXFLAGS=
    CUDA_LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L$CUDA_ROOT/lib64 -L$CUDA_ROOT/lib"
    CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
    LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"
#fi

# Defines
if test x$cuda = xtrue; then
  AC_DEFINE([THRUST_DEVICE_SYSTEM], [THRUST_DEVICE_SYSTEM_CUDA])
else
  AC_DEFINE([THRUST_DEVICE_SYSTEM], [THRUST_DEVICE_SYSTEM_OMP])
fi

# Checks for programs.
AC_PROG_CXX([icpc g++])
AC_PROG_CXXCPP([icpc g++])
AC_PROG_LIBTOOL
AC_PROG_INSTALL

if test x$CXX = xicpc; then
  intel=true
else
  intel=false
fi

# Compiler characteristics
AC_OPENMP

# Checks of programs
if test x$mpi = xtrue; then
    if test x$vampir = xtrue; then
	    AC_CHECK_PROG(have_mpicxxvt, [mpicxx-vt], [true], [false])
	    if test x$have_mpicxxvt = xfalse; then
	       AC_MSG_ERROR([program not found (required with --enable-vampir)])
	    fi
		CXX="mpicxx-vt --vt:cxx $CXX --vt:inst manual --vt:mpi"
		CXXCPP="mpicxx-vt --vt:cxx $CXXCPP --vt:mpi"
	else
	    AC_CHECK_PROG(have_mpicxx, [mpicxx], [true], [false])
	    if test x$have_mpicxx = xfalse; then
	       AC_MSG_ERROR([program not found (required with --enable-vampir)])
	    fi
		CXX="mpicxx"
		CXXCPP="mpicxx -E"
	fi
fi

# Checks for libraries
AC_CHECK_LIB([m], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([gfortran], [main], [], [AC_MSG_ERROR([required library not found])])

# Intel MKL if available, needing special treatment given multiple libs...
if test x$mkl = xtrue; then
	AC_SEARCH_LIBS([mkl_blas_dgemv], [mkl_intel mkl_intel_lp64], [], [], [-lmkl_sequential -lmkl_core])
	AC_CHECK_LIB([mkl_sequential], [mkl_blas_dgemv], [], [], [-lmkl_core])
	AC_CHECK_LIB([mkl_core], [mkl_blas_dgemv], [], [])
fi

# atlas if available, may be required later
AC_CHECK_LIB([atlas], [main], [], [])

# ...otherwise search for alternatives
AC_SEARCH_LIBS([cblas_dgemv], [openblas openblaso cblas gslcblas], [], [AC_MSG_ERROR([required library not found])])
AC_SEARCH_LIBS([dgemv_], [openblas openblaso blas f77blas], [], [AC_MSG_ERROR([required library not found])])
AC_SEARCH_LIBS([dpotrf_], [openblas openblaso lapack], [], [AC_MSG_ERROR([required library not found])])

AC_CHECK_LIB([qrupdate], [dch1dn_], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([gsl], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([netcdf], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([netcdf_c++], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([profiler], [main], [], [])

if test x$cuda = xtrue; then
    AC_CHECK_LIB([cuda], [main], [], [])
    AC_CHECK_LIB([cudart], [main], [], [AC_MSG_ERROR([library not found (required with --enable-cuda)])])
    AC_CHECK_LIB([curand], [main], [], [AC_MSG_ERROR([library not found (required with --enable-cuda)])])
    AC_CHECK_LIB([cublas], [main], [], [AC_MSG_ERROR([library not found (required with --enable-cuda)])])
    AC_CHECK_LIB([magmablas], [magmablas_dgemv_MLU], [], [AC_MSG_ERROR([library not found (required with --enable-cuda)])])
    AC_CHECK_LIB([magma], [magma_dpotrf], [], [AC_MSG_ERROR([library not found (required with --enable-cuda)])])
fi

if test x$mpi = xtrue; then
    AC_CHECK_LIB([mpi], [main], [], [AC_MSG_ERROR([library not found (required with --enable-mpi)])])
    AC_CHECK_LIB([boost_mpi], [main], [], [AC_MSG_ERROR([library not found (required with --enable-mpi)])])
    AC_CHECK_LIB([boost_serialization], [main], [], [AC_MSG_ERROR([library not found (required with --enable-mpi)])])
fi

# Checks for library functions
m4_foreach_w([func], [pow sqrt exp log], [AC_CHECK_FUNC(func, [], \
    AC_MSG_ERROR([required math function not found]))])

# Checks for header files
AC_HEADER_ASSERT

AC_CHECK_HEADER([netcdfcpp.h], [], \
    AC_MSG_ERROR([required NetCDF C++ interface header not found]), [])

if test x$mkl = xtrue; then
	AC_CHECK_HEADER([mkl_cblas.h], [], \
	    AC_MSG_ERROR([header not found (required with --enable-mkl)]), [-])
else
	AC_CHECK_HEADER([cblas.h], [], \
	    AC_MSG_ERROR([required header not found]), [-])
fi

m4_foreach_w([header_file], [\
    boost/mpl/if.hpp \
    boost/random/bernoulli_distribution.hpp \
    boost/random/gamma_distribution.hpp \
    boost/random/mersenne_twister.hpp \
    boost/random/normal_distribution.hpp \
    boost/random/uniform_int.hpp \
    boost/random/uniform_real.hpp \
    boost/random/variate_generator.hpp \
    boost/static_assert.hpp \
    boost/typeof/typeof.hpp \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([required boost header not found]), [])])

m4_foreach_w([header_file], [\
    thrust/adjacent_difference.h \
    thrust/binary_search.h \
    thrust/copy.h \
    thrust/device_ptr.h \
    thrust/distance.h \
    thrust/extrema.h \
    thrust/fill.h \
    thrust/for_each.h \
    thrust/functional.h \
    thrust/gather.h \
    thrust/inner_product.h \
    thrust/iterator/counting_iterator.h \
    thrust/iterator/detail/normal_iterator.h \
    thrust/iterator/discard_iterator.h \
    thrust/iterator/permutation_iterator.h \
    thrust/iterator/transform_iterator.h \
    thrust/iterator/zip_iterator.h \
    thrust/logical.h \
    thrust/reduce.h \
    thrust/scan.h \
    thrust/sequence.h \
    thrust/sort.h \
    thrust/transform.h \
    thrust/transform_reduce.h \
    thrust/transform_scan.h \
    thrust/tuple.h \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([required Thrust header not found]), [])])

m4_foreach_w([header_file], [\
    gsl/gsl_multimin.h \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([required GSL header not found]), [])])

if test x$cuda = xtrue; then
    AC_CHECK_HEADER([cublas_v2.h], [], AC_MSG_ERROR([CUBLAS header not found (required with --enable-cuda)]), [])
    AC_CHECK_HEADER([curand.h], [], AC_MSG_ERROR([CURAND header not found (required with --enable-cuda)]), [])

    m4_foreach_w([header_file], [\
    	magma_s.h \
    	magma_d.h \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([MAGMA header not found (required with --enable-cuda)]), [#include <magmablas.h>])])
fi

if test x$mpi = xtrue; then
    AC_CHECK_HEADER([mpi.h], [], [AC_MSG_ERROR([MPI header not found (required with --enable-mpi)])], [])
	AC_CHECK_HEADER([boost/mpi.hpp], [], [AC_MSG_ERROR([MPI header not found (required with --enable-mpi)])], [])
fi

# Restore user's flags
CPPFLAGS=$USER_CPPFLAGS
CXXFLAGS=$USER_CXXFLAGS
LDFLAGS=$USER_LDFLAGS

# Defines
AM_CONDITIONAL([ENABLE_SINGLE], [test x$single = xtrue])
AM_CONDITIONAL([ENABLE_CUDA], [test x$cuda = xtrue])
AM_CONDITIONAL([ENABLE_SSE], [test x$sse = xtrue])
AM_CONDITIONAL([ENABLE_MKL], [test x$mkl = xtrue])
AM_CONDITIONAL([ENABLE_MPI], [test x$mpi = xtrue])
AM_CONDITIONAL([ENABLE_VAMPIR], [test x$vampir = xtrue])
AM_CONDITIONAL([ENABLE_INTEL], [test x$intel = xtrue])

# Variables for automake
AC_SUBST([CUDA_CPPFLAGS], $CUDA_CPPFLAGS)
AC_SUBST([CUDA_CXXFLAGS], $CUDA_CXXFLAGS)
AC_SUBST([CUDA_LDFLAGS], $CUDA_LDFLAGS)

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
