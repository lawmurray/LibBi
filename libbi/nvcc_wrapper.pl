#!/usr/bin/perl

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
    if ($arg eq '-fPIC' || $arg eq '-DPIC') {
	push(@host_args, $arg);
    } else {
	push(@dev_args, $arg);
    }
}

$arg = '-Xcompiler="' . join(' ', @host_args) . '"';
splice(@dev_args, 1, 0, $arg);
$cmd = join(' ', @dev_args) . "\n";

exec($cmd) || die;
