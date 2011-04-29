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
    <www.boost.org/>,
  * sqlite3.

The following Perl modules are required:

  * Template,
  * Template::Plugin::DBI,
  * DBI,
  * DBD::SQLite,
  * Text::CSV,
  * List::MoreUtils Perl.

The following packages are optional:

  * Doxygen <www.doxygen.org> for building source code documentation,
  * dot,
  * graphviz,
  * pdflatex.


Compilation
------------------------------------------------------------------------------

To build the library, change into the lib/ directory:

cd lib

construct the Makefile by running:

./bootstrap.sh

and then compile with:

make -j

The following macros may be defined during compilation to adjust behaviour:

* USE_CONFIG=1 uses options specified in the config.mk file.
* DEBUG=1 builds in debug mode.
* RELEASE=1 builds in release mode.
* PROFILE=1 builds in profile mode.
* NDEBUG=1 will disable assertion checking.

* USE_CPU=1 will use CPU code, otherwise GPU code is used where possible.
* USE_SSE=1 will use SSE extensions for some host code.
* USE_DOUBLE=1 will use double precision arithmetic on the GPU, otherwise
  single precision is used.
* USE_FAST_MATH=1 will use intrinsic CUDA math functions for device code, as
  long as USE_DOUBLE is not defined (intrinsics are available only for single
  precision).
* USE_TEXTURE=1 will use textures for some device code, as long as USE_DOUBLE
  is not defined (textures support single precision only).
* USE_INTEL=1 uses icpc instead of gcc for compilation.
* USE_ODE=rk43|dopri5|rk4 sets the numerical integrator to use for ODE
  integration: low-storage adaptive RK4(3)5[2R+]C, adaptive DOPRI5(4) or
  classic fourth-order Runge-Kutta, respectively, with the first being the
  default.

* X_LEN=val sets the length of the x-dimension.
* Y_LEN=val sets the length of the y-dimension.
* Z_LEN=val sets the length of the z-dimension.

For example, a typical production build of GPU code, in double precision,
would use:

make USE_DOUBLE=1 RELEASE=1 NDEBUG=1 -j

or alternatively, these may be specified in config.mk, then using:

make USE_CONFIG=1 -j

Usage of config.mk is recommended. The procedure described below will copy
this file across to be used for compilation of model-specific generated code
also, ensuring compatibility between client code and library.

Note that USE_SSE=1 and USE_TEXTURE=1 will not necessarily improve, and may
degrade, performance, depending on the model. USE_FAST_MATH=1 will certainly
improve performance of device code, but may significantly degrade floating
point accuracy.


Usage
------------------------------------------------------------------------------

Create a new directory for your project. In that directory, create a data/
subdirectory and copy your NetCDF data files into it. Create a single <name>.csv
file containing your model specification, where <name> is some arbitrary name
that you have selected for your project.

The following command will generate all required files for your project:

<path to bi>/gen/bootstrap --model <name>.csv

All client programs can then be compiled with:

make -j

or individual client programs compile with, for example:

make -j build/simulate

The macros given above for compilation of the library may be used for this
make command also. Generally, you should compile with the same macros as given
for your bi library compilation, as different options may result in errors at
link time. The following additional macros may be used:

* USE_DOPRI5=1 will use the DOPRI5 integrator for ordinary differential
  equations, otherwise RK4(3)5[2R+]C is used.
* USE_MKL=1 uses Intel MKL instead of standard BLAS.
* USE_ATLAS=1 uses Atlas instead of standard BLAS.

After compilation, edit the config.sh file to specify your data files and
appropriate settings for your experiments.

Run any of the scripts in the scripts/ directory to perform the given
function, e.g.

scripts/simulate.sh

will run a simulation of the model, producing output in the results/
directory.

The bootstrap procedure should be repeated after making changes to
<name>.csv. The compilation procedure should be repeated after making changes
to config.mk. Changes to config.sh do not require additional steps before
running scripts again.

The following special make targets may be used:

make clean:   removes compiled binaries and artifacts
make clobber: removes all generated files


Documentation
------------------------------------------------------------------------------

Source code documentation for the library may be built using doxygen from
within the lib/ directory:

doxygen bi.doxyfile

HTML documentation will be produced in lib/docs/html.
