# Installation on Windows

LibBi can run under Cygwin on Windows. CUDA support has not been tested at this stage.

## Step 1: Install Cygwin and packages
 
Download the Cygwin setup program from <cygwin.org> and run it. During the installation you will be asked to select which packages to install. Choose at least the following:

* liblapack-devel
* libqrupdate-devel
* libboost-devel
* gsl-devel
* libnetcdf-devel
* mingw-gcc-g++
* autoconf
* automake
* libcrypt-devel
* curl
* wget
* perl

Note that, if you later find that you missed a package, simply run the installation program again and add the missing packages; Cygwin will update your install.

## Step 2: Install Thrust

Run Cygwin and enter the following at the command prompt:

    wget https://github.com/thrust/thrust/releases/download/v1.7.0/thrust-1.7.0.zip
    unzip thrust-1.7.0.zip
    sudo mv thrust /usr/local/include

## Step 5: Install LibBi

Install LibBi as per the instructions in `INSTALL.md`, running the commands in there from within Cygwin. You can then run LibBi from within Cygwin.

## Troubleshooting

### Errors while installing other Perl modules

This can be caused if your home directory has a space in its name. To fix this:

1. Exit Cygwin.
2. Work out where your Cygwin installation is, usually `c:\cygwin64`.
3. Find your home directory, usually in `c:\cygwin64\home`.
4. Change the name of your home directory so that it does not contain a space.
5. Remove the `.cpan` directory from within your home directory. This deletes your current `cpan` config, which may contain references to the old name.
6. Edit the `c:\cygwin64\etc\passwd` file. Find the line that pertains to your user account, and modify the path to your home directory there.
7. Restart Cygwin.

### FTP errors when installing other Perl modules

This can be fixed by ensuring that the `wget` package is install in Cygwin.
