AC_INIT([LibBi], [1.3.0], [bug-report@libbi.org], [libbi], [http://www.libbi.org])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Conditionals used by make
AC_ARG_ENABLE([assert],
     [  --enable-assert         enable assertion checking],
     [case "${enableval}" in
       yes) assert=true ;;
       no)  assert=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-assert]) ;;
     esac],[assert=true])

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

AC_ARG_ENABLE([cudafastmath],
     [  --enable-cuda-fast-math use fast-but-inaccurate math in CUDA],
     [case "${enableval}" in
       yes) cudafastmath=true ;;
       no)  cudafastmath=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-cuda-fast-math]) ;;
     esac],[cudafastmath=false])

AC_ARG_ENABLE([gpucache],
     [  --enable-gpucache       use GPU implementation of ancestry cache],
     [case "${enableval}" in
       yes) gpucache=true ;;
       no)  gpucache=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-gpucache]) ;;
     esac],[gpucache=false])

AC_ARG_ENABLE([sse],
     [  --enable-sse            use SSE code],
     [case "${enableval}" in
       yes) sse=true ;;
       no)  sse=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-sse]) ;;
     esac],[sse=false])

AC_ARG_ENABLE([avx],
     [  --enable-avx            use AVX code],
     [case "${enableval}" in
       yes) avx=true ;;
       no)  avx=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-avx]) ;;
     esac],[avx=false])

AC_ARG_ENABLE([openmp],
     [  --enable-openmp         use OpenMP multithreading],
     [case "${enableval}" in
       yes) openmp=true ;;
       no)  openmp=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-openmp]) ;;
     esac],[openmp=true])

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
     
AC_ARG_ENABLE([extradebug],
     [  --enable-extradebug    use extra debugging options],
     [case "${enableval}" in
       yes) extradebug=true ;;
       no)  extradebug=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-extradebug]) ;;
     esac],[extradebug=false])

AC_ARG_ENABLE([diagnostics],
     [  --enable-diagnostics    output diagnostics during run],
     [diagnostics="${enableval}"],[diagnostics=false])

AC_ARG_ENABLE([diagnostics2],
     [  --enable-diagnostics2    output diagnostics2 during run],
     [case "${enableval}" in
       yes) diagnostics2=true ;;
       no)  diagnostics2=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-diagnostics2]) ;;
     esac],[diagnostics2=false])


AC_ARG_ENABLE([gperftools],
     [  --enable-gperftools    enable gperftools profiler],
     [case "${enableval}" in
       yes) gperftools=true ;;
       no)  gperftools=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-gperftools]) ;;
     esac],[gperftools=false])
     
# Add standard CUDA directories
#if test x$cuda = xtrue; then
# ^ don't test, may be using Thrust bundled with CUDA, even for host
if test x$CUDA_ROOT != x; then
  CPPFLAGS="$CPPFLAGS -I$CUDA_ROOT/include -DHAVE_CUBLAS" # -DHAVE_CUBLAS needed for MAGMA 1.4
  LDFLAGS="$LDFLAGS -L$CUDA_ROOT/lib64 -L$CUDA_ROOT/lib -Wl,-rpath,$CUDA_ROOT/lib64 -Wl,-rpath,$CUDA_ROOT/lib"
fi
CPPFLAGS="$CPPFLAGS -I/usr/local/cuda/include"
LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -Wl,-rpath,/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib"
#fi

# Compilers etc
#
# On Mac OS X:
#   * clang++ supports SSE, AVX and CUDA, but not OpenMP
#   * custom g++ installs should support SSE, AVX and OpenMP, but not CUDA.
#     HomeBrew versions seem to produce extremely slow SSE and AVX code
#     though (this requires investigation).
#   * g++ symlinks to clang++ by default.
#
if test x$cuda = xtrue; then
	compilers="icpc clang++ g++"
else
	compilers="icpc g++ clang++"
fi

AC_PROG_CC([$compilers])
AC_PROG_CPP
AC_PROG_CXX([$compilers])
AC_PROG_CXXCPP
AC_PATH_PROG([NVCC], [nvcc], [nvcc], [$PATH:/usr/local/cuda/bin])
NVCC="$NVCC -ccbin=$CXX"
AC_PROG_INSTALL
AC_PROG_RANLIB

# Compiler characteristics
AC_CHECK_HEADERS([omp.h], [], [openmp=false], [-])
if test x$openmp = xtrue; then
  AC_OPENMP
fi

# Thrust backend
if test x$cuda = xtrue; then
  AC_DEFINE([THRUST_DEVICE_SYSTEM], [THRUST_DEVICE_SYSTEM_CUDA])
else
  AC_DEFINE([THRUST_DEVICE_SYSTEM], [THRUST_DEVICE_SYSTEM_CPP])
fi
if test x$openmp = xtrue; then
  AC_DEFINE([THRUST_HOST_SYSTEM], [THRUST_HOST_SYSTEM_OMP])
else
  AC_DEFINE([THRUST_HOST_SYSTEM], [THRUST_HOST_SYSTEM_CPP])
fi
# ^ OpenMP in Thrust 1.6 very slow, but seems to have been rectified in
#   Thrust 1.7, so its OpenMP backend has been re-enabled.

# Checks of programs
if test x$mpi = xtrue; then
    if test x$vampir = xtrue; then
	    AC_CHECK_PROG(have_mpicxxvt, [mpicxx-vt], [true], [false])
	    if test x$have_mpicxxvt = xfalse; then
	       AC_MSG_ERROR([Vampir compiler wrapper mpicxx-vt not found (only required with --enable-vampir)])
	    fi
		CXX="mpicxx-vt --vt:cxx $CXX --vt:inst manual --vt:mpi"
		NVCC="mpicxx-vt --vt:cxx $NVCC --vt:inst manual --vt:mpi"
		CXXCPP="mpicxx-vt --vt:cxx $CXXCPP --vt:mpi"
	else
	    AC_CHECK_PROG(have_mpicxx, [mpicxx], [true], [false])
	    if test x$have_mpicxx = xfalse; then
	       AC_MSG_ERROR([MPI compiler wrapper mpicxx not found (only required with --enable-mpi)])
	    fi
		CXXCPP="mpicxx -E"
		CXXFLAGS+=" `mpicxx -showme:compile`"
		LIBS+=" `mpicxx -showme:link`"
	fi
fi

# Checks for libraries
AC_CHECK_LIB([m], [main], [], [AC_MSG_ERROR([required standard math library not found])])
AC_CHECK_LIB([gfortran], [main], [], [])

# Intel MKL if available, needing special treatment given multiple libs...
AC_SEARCH_LIBS([mkl_blas_dgemv], [mkl_intel mkl_intel_lp64], [], [], [-lmkl_sequential -lmkl_core])
AC_CHECK_LIB([mkl_sequential], [mkl_blas_dgemv], [], [], [-lmkl_core])
AC_CHECK_LIB([mkl_core], [mkl_blas_dgemv], [], [])

# atlas if available, may be required later
AC_CHECK_LIB([atlas], [main], [], [])

# ...otherwise search for alternatives
AC_SEARCH_LIBS([cblas_dgemv], [openblaso openblas cblas gslcblas], [], [AC_MSG_ERROR([required CBLAS library not found])])
AC_SEARCH_LIBS([dgemv_], [openblaso openblas blas f77blas], [], [AC_MSG_ERROR([required BLAS library not found])])
AC_SEARCH_LIBS([dpotrf_], [openblaso openblas lapack], [], [AC_MSG_ERROR([required LAPACK library not found])])

AC_CHECK_LIB([qrupdate], [dch1dn_], [], [AC_MSG_ERROR([required QRUpdate library not found])])
AC_CHECK_LIB([gsl], [main], [], [AC_MSG_ERROR([required GSL library not found])])
AC_CHECK_LIB([netcdf], [main], [], [AC_MSG_ERROR([required NetCDF library not found])])
AC_CHECK_LIB([profiler], [main], [], [])

if test x$cuda = xtrue; then
    AC_CHECK_LIB([cuda], [main], [], [])
    AC_CHECK_LIB([cudart], [main], [], [AC_MSG_ERROR([CUDA library not found (only required with --enable-cuda)])])
    AC_CHECK_LIB([curand], [main], [], [AC_MSG_ERROR([CURAND library (part of CUDA) not found (only required with --enable-cuda)])])
    AC_CHECK_LIB([cublas], [main], [], [AC_MSG_ERROR([CUBLAS library (part of CUDA) not found (only required with --enable-cuda)])])
fi

if test x$mpi = xtrue; then
    AC_CHECK_LIB([mpi], [main], [], [AC_MSG_ERROR([MPI library not found (only required with --enable-mpi)])])
    AC_CHECK_LIB([boost_mpi], [main], [], [AC_MSG_ERROR([Boost.MPI library not found (only required with --enable-mpi)])])
    AC_CHECK_LIB([boost_serialization], [main], [], [AC_MSG_ERROR([Boost.Serialization library not found (only required with --enable-mpi)])])
fi

# Checks for library functions
m4_foreach_w([func], [pow sqrt exp log], [AC_CHECK_FUNC(func, [], \
    AC_MSG_ERROR([required standard math function not found]))])

# Checks for header files
AC_HEADER_ASSERT

AC_CHECK_HEADERS([netcdf.h], [], \
    AC_MSG_ERROR([required NetCDF header not found]), [-])

AC_CHECK_HEADERS([mkl_cblas.h cblas.h gsl/gsl_cblas.h], [], [], [-])
if test x$ac_cv_header_mkl_cblas_h = xfalse && test x$ac_cv_header_cblas_h = xfalse && x$ac_cv_header_gsl_gsl_cblas_h = xfalse; then
    AC_MSG_ERROR([required CBLAS header not found])
fi

AC_CHECK_HEADERS([\
    boost/mpl/if.hpp \
    boost/random/binomial_distribution.hpp \
    boost/random/bernoulli_distribution.hpp \
    boost/random/gamma_distribution.hpp \
    boost/random/mersenne_twister.hpp \
    boost/random/normal_distribution.hpp \
    boost/random/poisson_distribution.hpp \
    boost/random/uniform_int.hpp \
    boost/random/uniform_real.hpp \
    boost/random/variate_generator.hpp \
    boost/typeof/typeof.hpp \
    ], [], AC_MSG_ERROR([required Boost header not found]), [-])

AC_CHECK_HEADERS([\
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
    ], [], AC_MSG_ERROR([required Thrust header not found]), [-])

AC_CHECK_HEADERS([gsl/gsl_multimin.h], [], AC_MSG_ERROR([required GSL header not found]), [-])

if test x$cuda = xtrue; then
    AC_CHECK_HEADERS([cublas_v2.h], [], AC_MSG_ERROR([CUBLAS header not found (only required with --enable-cuda)]), [-])
    AC_CHECK_HEADERS([curand.h], [], AC_MSG_ERROR([CURAND header not found (only required with --enable-cuda)]), [-])
fi

if test x$mpi = xtrue; then
    AC_CHECK_HEADERS([mpi.h], [], [AC_MSG_ERROR([MPI header not found (only required with --enable-mpi)])], [])
	AC_CHECK_HEADERS([boost/mpi.hpp], [], [AC_MSG_ERROR([Boost.MPI header not found (only required with --enable-mpi)])], [])
fi

if test x$gperftools = xtrue; then
    AC_CHECK_HEADERS([google/profiler.h], [], [AC_MSG_ERROR([Gperftools header not found (only required with --enable-gperftools)])], [])
fi

# Defines
AM_CONDITIONAL([ENABLE_ASSERT], [test x$assert = xtrue])
AM_CONDITIONAL([ENABLE_SINGLE], [test x$single = xtrue])
AM_CONDITIONAL([ENABLE_CUDA], [test x$cuda = xtrue])
AM_CONDITIONAL([ENABLE_CUDA_FAST_MATH], [test x$cudafastmath = xtrue])
AM_CONDITIONAL([ENABLE_GPU_CACHE], [test x$gpucache = xtrue])
AM_CONDITIONAL([ENABLE_SSE], [test x$sse = xtrue])
AM_CONDITIONAL([ENABLE_AVX], [test x$avx = xtrue])
AM_CONDITIONAL([ENABLE_OPENMP], [test x$openmp = xtrue])
AM_CONDITIONAL([ENABLE_MPI], [test x$mpi = xtrue])
AM_CONDITIONAL([ENABLE_VAMPIR], [test x$vampir = xtrue])
AM_CONDITIONAL([ENABLE_EXTRADEBUG], [test x$extradebug = xtrue])
AM_CONDITIONAL([ENABLE_GPERFTOOLS], [test x$gperftools = xtrue])

AC_DEFINE_UNQUOTED([ENABLE_DIAGNOSTICS], [$diagnostics])

# Variables for automake
#AC_SUBST([NVCC], $NVCC)

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
