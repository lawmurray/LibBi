# LibBi Installation

LibBi runs on Linux, Mac OS X, and Windows under Cygwin. Installation of LibBi itself is straighforward, but the C++ code which LibBi generates, compiles and executes has a number of dependencies that will also need to be installed.

This file contains the instructions for installing LibBi itself on any platform. To install dependencies, see the walkthroughts for each platform in `INSTALL_LINUX.md`, `INSTALL_MAC.md` and `INSTALL_WIN.md`. For other platforms, or greater control over the installation, `INSTALL_GENERIC.md` provides all the details. It is recommended that you do all this first, as the LibBi installation below can then test to ensure that all dependencies are satisfied.

## The Quick Way

The quickest way to install LibBi is with the `cpan` utility. This will automatically install all of the Perl modules on which LibBi depends, as well as LibBi itself. From within the LibBi directory, run:

    sudo cpan .
    
for a system-wide install, or

    cpan .
    
for a user-local install.
    
If this is the first time that you have run `cpan`, it will ask you a few questions while it configures itself, and may then ask you to restart your shell and run it again before continuing.

## The More Involved Way

LibBi's dependencies can be installed by running `cpan` and then entering:

    install Template Graph Math::Symbolic Carp::Assert Parse::Yapp Parse::Lex File::Slurp File::ShareDir Getopt::ArgvFile

Exit `cpan` by typing `quit`. LibBi can then be installed (or upgraded) by issuing the following sequence of commands from within the LibBi directory:

    perl Makefile.PL
    make
    make test
    sudo make install

## The Development Way

If you are developing LibBi, or otherwise have a need to use the latest development version, the following set up is recommended so that you can pull down updates without having to reinstall.

Firstly, clone the GitHub repository:

    git clone https://github.com/libbi/LibBi.git
    
Now, you can ensure that this version is used when running LibBi by using the full path to the LibBi/script/libbi program, e.g.

    ~/LibBi/script/libbi sample @config.conf @posterior.conf
    
or, as this rapidly gets cumbersome, you can add the full path of the `LibBi/script` directory to your `PATH` environment variable. How to do this will depend on which shell you are using. For `bash` it usually means editing your `~/.bashrc` or `~/.bash_profile` file. On Mac OS X it will mean editing your `~/.profile` file.

Once you have done this, type `which libbi` to confirm that the version of LibBi being used is that which you expect. Henceforth you can then type `libbi` to run.

Updates to the development version can be pulled down from the GitHub repository with:

    git pull origin master
    
They will take effect the next time that you run `libbi`, with no installation required.
