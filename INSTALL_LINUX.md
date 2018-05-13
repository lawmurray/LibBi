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

## Step 3: Install LibBi

### Option 1: Using Linuxbrew

First install the Linuxbrew package manager using the instructions
[here](http://linuxbrew.sh). Then simply run the following command from a terminal
window to install LibBi: 

    brew install libbi

### Option 2: Manual install

#### Step 3a: Install Thrust

If CUDA *is not* installed, it is necessary to install Thrust (this is not
necessary if using the Linuxbrew route described above, as this will
automatically download Thrust). If CUDA *is* installed, Thrust is already
included with it, and need not be installed separately. 

To install Thrust:

    wget https://github.com/thrust/thrust/releases/download/1.8.2/thrust-1.8.2.zip
    unzip thrust-1.8.2.zip
    sudo mv thrust /usr/local/include

## Step 3b: Install LibBi

Install LibBi manually as per the instructions in `INSTALL.md`.
