#!/usr/bin/env perl

##
## Wrapper around nvcc for compatibility with libtool.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

my @host_args;
my @dev_args;
my $arg;
my $cmd;

foreach $arg (@ARGV) {
    $arg =~ s/(-pthread)/-Xcompiler="$1"/g;
    $arg =~ s/-g\d+/-g/g;
    $arg =~ s/(-f(?:instrument-functions|unroll-loops|no-inline|PIC))/-Xcompiler="$1"/g;
    $arg =~ s/(-D(?:PIC))/-Xcompiler="$1"/g;

    push(@dev_args, $arg);
}

$cmd = join(' ', @dev_args) . "\n";

exec($cmd) || die;
