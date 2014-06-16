# Installation on Mac OS X

LibBi can be installed on Mac OS X by using Apple's Xcode and a
package manager such as Homebrew or MacPorts. The recommended package
manager is Homebrew---we find that it "just works" with no additional
configuration required---but instructions for MacPorts are included
below also.

## Step 1: Install Xcode
 
The easiest way to install Xcode is via the Mac App Store. Once
installed, launch it and use the *Xcode > Preferences* menu item to
install the *Command Line Tools* package. This will give you some of
the basics required by LibBi.

## Step 2 (optional): Install CUDA

If you have a CUDA-enabled GPU and would like LibBi to be able to use
it, you must also download and install the CUDA Toolkit
<https://developer.nvidia.com/cuda-toolkit>.

## Step 3: Install Thrust

If CUDA *is not* installed, it is necessary to install Thrust. If CUDA
*is* installed, Thrust is already included with it, and need not be
installed separately.

To install Thrust:

    wget https://github.com/thrust/thrust/releases/download/v1.7.0/thrust-1.7.0.zip
    unzip thrust-1.7.0.zip
    sudo mv thrust /usr/local/include

## Step 4: Install Homebrew and dependencies

Open Terminal and install Homebrew (<brew.sh>) with the following command:

    ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/Homebrew/go/install)"

This may give you additional instructions to follow to ensure that you
end up with a working environment.

Install the Homebrew packages on which LibBi depends:

    brew tap homebrew/science
    brew install qrupdate netcdf gsl boost
     
## Step 5: Install LibBi

Install LibBi as per the instructions in `INSTALL.md`.

## Alternative step 4: Install MacPorts and dependencies

LibBi has also been successfully installed with MacPorts instead of
Homebrew. This seems more difficult, in that (a) it is necessary to
edit environment variables in your `~/.profile` file, and (b) it can
take much longer to install dependencies. Nevertheless MacPorts may be
preferable, especially if you already have a working installation of
it.

Install the MacPorts packages on which LibBi depends:

    sudo port install autoconf automake netcdf qrupdate atlas boost gsl

To ensure that LibBi can find the `gfortran` libraries, symlink them
into the `/opt/local/lib` directory:

    sudo ln -s /opt/local/lib/gcc45/libgfortran.a /opt/local/lib/.
    sudo ln -s /opt/local/lib/gcc45/libgfortran.dylib /opt/local/lib/.

Now set up the environment for LibBi to compile its client
programs. Firstly, add the following to `~/.profile`

    export LDFLAGS='-L/opt/local/lib'
    export CPPFLAGS='-I/opt/local/include'
    export LD_LIBRARY_PATH=/opt/local/lib
    export LD_INCLUDE_PATH=/opt/local/include

LibBi needs to use the MacPorts `g++` compiler in `/opt/local/bin`,
not the system default `g++` in `/usr/bin`. To do this, set the
following environment variable:

    export CXX=/opt/local/bin/g++-mp-*

You can put this in `~/.profile` too, although keep in mind that all
configure scripts that you run will then use this `g++`, not just
LibBi. This may not be a problem. If it is, just enter the above line
each time you open a Terminal window.

Quit Terminal then start it again, to ensure that these environment
variables are set.
