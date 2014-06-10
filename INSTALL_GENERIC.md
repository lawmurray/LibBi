# Generic installation

The suggested configurations for Linux, Mac OS X and Windows are given in the `INSTALL_LINUX.md`, `INSTALL_MAC.md` and `INSTALL_WIN.md` files, respectively. This file provides additional detail for other platforms or custom installs.

## Requirements

The following packages are required:

  * a C++ compiler,
  * [GNU Autotools](http://www.gnu.org/software/): autoconf and automake,
  * [Thrust](http://thrust.github.com) 1.6 or later,
  * [NetCDF](http://www.unidata.ucar.edu/software/netcdf/) 4.2 or later,
  * [QRUpdate](http://qrupdate.sourceforge.net),
  * [GNU Scientific Library](http://www.gnu.org/software/gsl/),
  * [Boost](http://www.boost.org) 1.43 or later, specifically Boost.Random and
    Boost.TypeOf,
  * a [BLAS](http://www.netlib.org/blas/) implementation,
  * a [LAPACK](www.netlib.org/lapack/) implementation.

Note that you should install, where applicable, both the compiled libraries
and headers of these libraries; on some Linux distributions the headers are
contained in separate packages of the same name with `-devel` appended.

For GPU support, the following additional packages are required:

  * [CUDA](http://www.nvidia.com/cuda) 5.0 or later (including CUBLAS and
    CURAND),
  * [MAGMA](icl.cs.utk.edu/magma/) 1.2.1 or later.

The latter can be difficult to install and for almost everything you might like to do with LibBi, is not required. You may like to do without it.

For MPI support, the following additional packages are required:

  * An MPI implementation,
  * Boost.MPI and Boost.Serialization.

The following packages are optional for building documentation and visualising models:

  * [Doxygen](http://www.doxygen.org),
  * dot,
  * graphviz,
  * pod2latex,
  * latex2html,
  * pdflatex.
