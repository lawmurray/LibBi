# LibBi Installation

LibBi runs on Linux, Mac OS X, and Windows under Cygwin. Installation
of LibBi itself is straighforward, but LibBi works by generating,
compiling and running C++ code; this code has a number of dependencies
that will also need to be installed.

This file contains the instructions for installing LibBi itself. This
process is the same on all platforms. To install dependencies of the
C++ code that LibBi generates, see the walkthroughts for each platform
in `INSTALL_LINUX.md`, `INSTALL_MAC.md` and `INSTALL_WIN.md` for Linux
(and possibly other Unix), Mac OS X and Windows, respectively. For
other platforms, or for a custom install, see `INSTALL_GENERIC.md`.

You should install all dependencies before following the instructions
below, as the last step performs some basic tests on them.

## The Quick Way

LibBi itself is just a Perl module. The quickest way to install it is
with the `cpan` utility. This will also install some extra Perl
modules on which LibBi depends. From within the LibBi directory, run:

    sudo cpan .

If this is the first time that you have run `cpan`, it will ask a few
questions to configure itself.

If you do not have root privileges, you can instead run just `cpan .`
to install LibBi in your home directory. If this is the first time you
have run `cpan` in this way, it may ask you to restart your shell
after it has configured itself---keep an eye out for this.

## The More Involved Way

LibBi's dependencies can be installed by running `sudo cpan` and then
entering:

    install Template Graph Math::Symbolic Carp::Assert Parse::Yapp \
        Parse::Lex File::Slurp File::ShareDir Getopt::ArgvFile

Exit `cpan` by typing `quit`. LibBi can then be installed (or
upgraded) by issuing the following sequence of commands from within
the LibBi directory:

    perl Makefile.PL
    make
    make test
    sudo make install

## The Development Way

If you are developing LibBi, or otherwise have a need to use the
latest development version, the following set up is recommended so
that you can pull down updates without having to reinstall.

Firstly, clone the GitHub repository:

    git clone https://github.com/libbi/LibBi.git

You can use the full path to the `LibBi/script/libbi` program when
running LibBi to ensure that this version is used, e.g.

    ~/LibBi/script/libbi sample @config.conf @posterior.conf
    
or, as this rapidly gets cumbersome, you can add the full path of the
`LibBi/script` directory to your `PATH` environment variable. How to
do this will depend on which shell you are using. For `bash` it
usually means editing your `~/.bashrc` or `~/.bash_profile` file. On
Mac OS X it will mean editing your `~/.profile` file.

After changing your `PATH` environment variable, restart your shell
and type `which libbi` to confirm that the correct version of LibBi is
associated with the command `libbi`. Henceforth you can then just type
`libbi` to run.

Updates to the development version can be pulled down from the GitHub
repository at any time with:

    git pull origin master

They will take effect the next time that you run `libbi`, with no
installation required.
