bi (Bayesian Inference) C++ library
------------------------------------------------------------------------------

Requirements
------------------------------------------------------------------------------

The following packages are required by the library:

  * CUDA 3.2 or later <www.nvidia.com/cuda>,
  * Thrust 1.3 or later <thrust.googlecode.com>,
  * MAGMA 0.2 or later <icl.cs.utk.edu/magma/>,
  * a BLAS implementation <www.netlib.org/blas/>,
  * a LAPACK implementation <www.netlib.org/lapack/>,
  * NetCDF 4.0.1 or later <www.unidata.ucar.edu/software/netcdf/>,
  * Boost 1.43 or later, specifically Boost.Random and Boost.TypeOf
    <www.boost.org/>.

The following (non-standard) Perl modules are required:

  * Template,
  * Math::Symbolic,
  * Carp::Assert,
  * Parse::Yapp,
  * Parse::Lex,
  * Getopt::ArgvFile.

The following packages are optional:

  * Doxygen <www.doxygen.org> for building source code documentation,
  * dot,
  * graphviz,
  * pdflatex.


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
