LibBi
------------------------------------------------------------------------------

Library component of the Bi ("Bayesian inference") package.


Requirements
------------------------------------------------------------------------------

The following packages are required by the library:

  * Thrust 1.6 or later <thrust.googlecode.com>,
  * NetCDF 4.0.1 or later <www.unidata.ucar.edu/software/netcdf/>,
  * QRUpdate <qrupdate.sourceforge.net/>,
  * GNU Scientific Library <www.gnu.org/software/gsl/>,
  * Boost 1.43 or later, specifically Boost.Random and Boost.TypeOf
    <www.boost.org/>,
  * a BLAS implementation <www.netlib.org/blas/>,
  * a LAPACK implementation <www.netlib.org/lapack/>.

Note that you should install, where applicable, both the compiled libraries
and headers of these libraries; often the headers are available via separate
*-devel packages in the package manager of your Linux distribution.

For GPU support, the following additional packages are required:

  * CUDA 4.0 or later (including CUBLAS and CURAND) <www.nvidia.com/cuda>,
  * MAGMA 0.2 or later <icl.cs.utk.edu/magma/>.

The following packages are optional for building documentation:

  * Doxygen <www.doxygen.org> for building source code documentation.
  * dot,
  * graphviz.


Compilation
------------------------------------------------------------------------------

If you have obtained LibBi from a source-code repository, use the following
first to generate a configure script via GNU Autoconf:

./autogen.sh

To build the library, use:

./configure CXXFLAGS="-03"
make -j 4
make install

The following additional options to configure are recommended, where
applicable:

* --enable-gpu, enable CUDA-based GPU support,
* --enable-sse, enable SSE instruction use.

If you wish to use single-precision floating point for your models rather than
double precision, you will need to build the single precision version of the
library. This is enabled with the following configure option:

* --disable-double, use single precision floating point.


Documentation
------------------------------------------------------------------------------

Source code documentation for the library may be built using doxygen with the
following command:

doxygen

HTML documentation can then be accessed at docs/html/index.html.
