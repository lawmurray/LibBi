# Installation on Linux

On Linux, dependencies are readily available from package
managers. See the list of requirements in `INSTALL_GENERIC.md` for
what to install, or the specific instructions for Ubuntu and openSUSE
below.

## Step 1: Install required packages

On Ubuntu:

    sudo apt-get install libblas-dev liblapack-dev libqrupdate-dev \
        libboost-all-dev libgsl0-dev libnetcdf-dev autoconf automake

On openSUSE:

    sudo zypper install blas lapack qrupdate boost-devel gsl-devel \
        netcdf-devel autoconf automake

## Step 2 (optional): Install CUDA

If you have a CUDA-enabled GPU and would like LibBi to be able to use
it, you must also download and install the CUDA Toolkit
<https://developer.nvidia.com/cuda-toolkit>.

## Step 3: Install Thrust

If CUDA *is not* installed, it is necessary to install Thrust. If CUDA
*is* installed, Thrust is included with it, and need not be installed
separately.

To install Thrust:

    wget https://github.com/thrust/thrust/releases/download/v1.7.0/thrust-1.7.0.zip
    unzip thrust-1.7.0.zip
    sudo mv thrust /usr/local/include

## Step 4: Install LibBi

Install LibBi as per the instructions in `INSTALL.md`.
