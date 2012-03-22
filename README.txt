bi (Bayesian Inference) C++ library
------------------------------------------------------------------------------

Requirements
------------------------------------------------------------------------------

The following packages are required by the library:

  * Thrust 1.3 or later <thrust.googlecode.com>,
  * NetCDF 4.0.1 or later <www.unidata.ucar.edu/software/netcdf/>,
  * qrupdate <qrupdate.sourceforge.net/>,
  * Boost 1.43 or later, specifically Boost.Random and Boost.TypeOf
    <www.boost.org/>,
  * a BLAS implementation <www.netlib.org/blas/>,
  * a LAPACK implementation <www.netlib.org/lapack/>.

For GPU support, the following additional packages are required:

  * CUDA 4.0 or later (including CUBLAS and CURAND) <www.nvidia.com/cuda>,
  * MAGMA 0.2 or later <icl.cs.utk.edu/magma/>.

The following packages are optional for building documentation and visualising
models:

  * Doxygen <www.doxygen.org> for building source code documentation,
  * dot,
  * graphviz,
  * pdflatex.

The following (non-standard) Perl modules are required:

  * Template,
  * Math::Symbolic,
  * Carp::Assert,
  * Parse::Yapp,
  * Parse::Lex,
  * Getopt::ArgvFile.


Compilation
------------------------------------------------------------------------------

To build the library, change into the lib/ directory:

cd lib

then compile with:

./configure
make -j 4
make install


Documentation
------------------------------------------------------------------------------

Source code documentation for the library may be built using doxygen from
within the lib/ directory:

doxygen bi.doxyfile

HTML documentation will be produced in lib/docs/html.
