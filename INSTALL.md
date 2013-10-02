LibBi INSTALL
=============

LibBi runs on Linux, and on Mac OS X with dependencies installed from the
[MacPorts](http://www.macports.org) project. It has not been tried on Windows;
[Cygwin](http://www.cygwin.com) would be a good bet, however, and if you do
get it running there the contribution of installation instructions would be
appreciated.

LibBi itself simply installs as a Perl module. The trick, however, is that the
C++ code that it generates to run your models has a number of dependencies. If
these are not installed you will get errors much like this the first time you
run LibBi:

    Error: ./configure failed with return code 1, see
    .Model/build_assert/configure.log and .Model/build_assert/config.log for
    details

If this occurs you should bring up the `configure.log` file mentioned. This is
the output of the usual `configure` script run when building software with a
GNU Autotools build system. You will be able to determine the missing
dependency from there.


Requirements
------------

The following non-standard Perl modules are required:

* `Template`
* `Graph`
* `Math::Symbolic`
* `Carp::Assert`
* `Parse::Yapp`
* `Parse::Lex`
* `File::Slurp`
* `File::ShareDir`
* `Getopt::ArgvFile`

These can be installed using the `cpan` program available on most systems.
After running `cpan`, simply type:

    install Template Graph Math::Symbolic ... etc

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

For MPI support, the following additional packages are required:

  * An MPI implementation,
  * Boost.MPI and Boost.Serialization.

The following packages are optional for building documentation and visualising
models:

  * [Doxygen](http://www.doxygen.org),
  * dot,
  * graphviz,
  * pod2latex,
  * latex2html,
  * pdflatex.


Generic installation instructions
---------------------------------

Once requirements are installed, LibBi is simply installed as a Perl
module. Use the following sequence:

    perl Makefile.PL
    make
    make test
    make install

This will install LibBi for system-wide usage.

Alternatively, keep LibBi in a local directory, or clone the Git repository,
and run it directly from that location with:

    /path/to/libbi/script/libbi ...

You can also add `/path/to/libbi/script` to your `PATH` so that you can just
use the following from any directory:

    libbi ...

Setting up in a local directory like this might be preferred if you are using
the development version of LibBi.


Specific installation instructions for Ubuntu Linux
---------------------------------------------------

Install required packages:

    sudo apt-get install liblapack-dev libblas-dev libgsl0-dev gsl-bin \
        octave-gsl octave-statistics libqrupdate-dev libboost-all-dev \
        libnetcdf-dev ncview netcdf-bin octave-octcdf doxygen graphviz dot2tex \
        latex2html libtool libgsl0-dbg

Download Thrust from thrust.github.com, unzip and move the thrust/ directory
to /usr/local/include/thrust.

Install Perl modules:

    sudo cpan

then within cpan:

    install Template Graph Math::Symbolic Carp::Assert Parse::Yapp Parse::Lex \
        File::Slurp File::ShareDir Getopt::ArgvFile

Finally, install LibBi as in the generic instructions above.


Specific installation instructions for Mac OS X
-----------------------------------------------

First ensure that you have Xcode installed. Launch it and use its *Xcode >
Preferences* menu item to install the *Command Line Tools* package. This will
give you some of the basics required by LibBi.

The remaining requirements can be satisfied from the MacPorts Project
<macports.org/>. The following step seems necessary on Mac OS X 10.8 (Mountain
Lion) or some MacPorts packages do not install:

    sudo xcodebuild -license

Now install required packages from MacPorts:

    sudo port install autoconf automake netcdf netcdf-cxx qrupdate atlas \
        boost gsl

This can take over an hour!

To ensure that LibBi can find the gfortran libraries, symlink them into the
/opt/local/lib directory as follows:

    sudo ln -s /opt/local/lib/gcc45/libgfortran.a /opt/local/lib/.
    sudo ln -s /opt/local/lib/gcc45/libgfortran.dylib /opt/local/lib/.

Download Thrust <thrust.github.com/>, unzip and move the thrust/ directory
to /opt/local/include/:

    sudo mv thrust /opt/local/include/.

Now install the required Perl modules:

    sudo cpan

then within cpan:

    install Template Graph Math::Symbolic Carp::Assert Parse::Yapp Parse::Lex \
        File::Slurp File::ShareDir Getopt::ArgvFile

It is important that you do this *after* installing the MacPorts dependencies
above, otherwise the modules will be installed to the system default perl, not
MacPorts' perl.

Now, set up the environment for LibBi to compile its client programs. Firstly,
add the following to ~/.profile

    export LDFLAGS='-L/opt/local/lib'
    export CPPFLAGS='-I/opt/local/include'
    export LD_LIBRARY_PATH=/opt/local/lib
    export LD_INCLUDE_PATH=/opt/local/include

LibBi needs to use the MacPorts `g++` compiler in `/opt/local/bin`, not the
system default `g++` in `/usr/bin`. To do this, set the following environment
variable:

    export CXX=/opt/local/bin/g++-mp-*

You can put this in `~/.profile` too, although keep in mind that all configure
scripts that you run will then use this `g++`, not just LibBi. This may not be
a problem. If it is, just enter the above line each time you open a Terminal
window.

Quit Terminal then start it again, to ensure that these environment variables
are set.

Finally, install LibBi as per the generic instructions above. To modify your
`PATH` variable, edit your `~/.profile` file.
