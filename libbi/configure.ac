AC_INIT([bi], [0.0.0], [bug-report@example.com], [bi], [http://www.example.com])
AC_PREREQ([2.59])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Conditionals used by make
AC_ARG_ENABLE([double],
     [  --enable-double         use double-precision floating point],
     [case "${enableval}" in
       yes) double=true ;;
       no)  double=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-double]) ;;
     esac],[double=true])
AM_CONDITIONAL([ENABLE_DOUBLE], [test x$double = xtrue])

AC_ARG_ENABLE([gpu],
     [  --enable-gpu            use CUDA-enabled GPU device],
     [case "${enableval}" in
       yes) gpu=true ;;
       no)  gpu=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-gpu]) ;;
     esac],[gpu=false])
AM_CONDITIONAL([ENABLE_GPU], [test x$gpu = xtrue])

AC_ARG_ENABLE([sse],
     [  --enable-sse            use SSE intrinsics],
     [case "${enableval}" in
       yes) sse=true ;;
       no)  sse=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-sse]) ;;
     esac],[sse=false])
AM_CONDITIONAL([ENABLE_SSE], [test x$sse = xtrue])

# Add standard CUDA directories
#if test x$gpu = xtrue; then
	CPPFLAGS="$CPPFLAGS -I/usr/local/cuda/include"
	LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib"
#fi

# Defines
AC_DEFINE([THRUST_DEVICE_SYSTEM], [THRUST_DEVICE_SYSTEM_OMP])

# Compilers
if test x$gpu = xtrue; then
  compilers="g++" # don't permit icpc, OpenMP compatibility issues
else
  compilers="icpc g++"
fi

# Checks for programs.
AC_PROG_CXX([$compilers])
AC_PROG_LIBTOOL
AC_PROG_INSTALL

# Compiler characteristics
AC_OPENMP

# Checks of programs
#AC_CHECK_PROG(have_ncconfig, [nc-config], [yes], [no])
#if test "$have_ncconfig" != "yes"; then
#  AC_MSG_ERROR([nc-config required])
#fi
#AC_SUBST(DEPS_LIBS, [`nc-config --libs`])

# Checks for libraries
AC_CHECK_LIB([m], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([gfortran], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([atlas], [main], [], []) # may be required if ATLAS cblas succeeds below
AC_SEARCH_LIBS([cblas_dgemv], [openblas openblaso cblas gslcblas], [], [AC_MSG_ERROR([required library not found])])
AC_SEARCH_LIBS([dgemv_], [openblas openblaso blas f77blas], [], [AC_MSG_ERROR([required library not found])])
AC_SEARCH_LIBS([dpotrf_], [openblas openblaso lapack], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([qrupdate], [dch1dn_], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([gsl], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([netcdf], [main], [], [AC_MSG_ERROR([required library not found])])
AC_CHECK_LIB([netcdf_c++], [main], [], [AC_MSG_ERROR([required library not found])])

if test x$gpu = xtrue; then
	AC_CHECK_LIB([cuda], [main], [], [])
	AC_CHECK_LIB([cudart], [main], [], [AC_MSG_ERROR([library not found (required with --enable-gpu)])])
	AC_CHECK_LIB([curand], [main], [], [AC_MSG_ERROR([library not found (required with --enable-gpu)])])
	AC_CHECK_LIB([cublas], [main], [], [AC_MSG_ERROR([library not found (required with --enable-gpu)])])
	AC_CHECK_LIB([magmablas], [magmablas_dgemv_MLU], [], [AC_MSG_ERROR([library not found (required with --enable-gpu)])])
	AC_CHECK_LIB([magma], [magma_dpotrf], [], [AC_MSG_ERROR([library not found (required with --enable-gpu)])])
fi

# Checks for library functions
m4_foreach_w([func], [pow sqrt], [AC_CHECK_FUNC(func, [], \
    AC_MSG_ERROR([required math function not found]))])

# Checks for header files
AC_HEADER_ASSERT

AC_CHECK_HEADER([netcdfcpp.h], [], \
    AC_MSG_ERROR([required NetCDF C++ interface header not found]), [-])

AC_CHECK_HEADER([cblas.h], [], \
    AC_MSG_ERROR([required CBLAS header not found]), [-])
    
m4_foreach_w([header_file], [\
    boost/mpl/if.hpp \
    boost/random/bernoulli_distribution.hpp \
    boost/random/gamma_distribution.hpp \
    boost/random/mersenne_twister.hpp \
    boost/random/normal_distribution.hpp \
    boost/random/uniform_int.hpp \
    boost/random/uniform_real.hpp \
    boost/random/variate_generator.hpp \
    boost/serialization/base_object.hpp \
    boost/serialization/serialization.hpp \
    boost/serialization/set.hpp \
    boost/serialization/split_member.hpp \
    boost/serialization/vector.hpp \
    boost/static_assert.hpp \
    boost/typeof/typeof.hpp \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([required boost header not found]), [-])])

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
    AC_MSG_ERROR([required Thrust header not found]), [-])])

m4_foreach_w([header_file], [\
    gsl/gsl_multimin.h \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([required GSL header not found]), [-])])

if test x$gpu = xtrue; then
	AC_CHECK_HEADER([cublas_v2.h], [], AC_MSG_ERROR([CUBLAS header not found (required with --enable-gpu)]), [-])
	AC_CHECK_HEADER([curand.h], [], AC_MSG_ERROR([CURAND header not found (required with --enable-gpu)]), [-])

	m4_foreach_w([header_file], [\
    	auxiliary.h \
    	magma_s.h \
    	magma_d.h \
    ], [AC_CHECK_HEADER(header_file, [], \
    AC_MSG_ERROR([MAGMA header not found (required with --enable-gpu)]), [-])])
fi

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
