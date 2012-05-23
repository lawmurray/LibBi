=head1 NAME

simulate - frontend to simulation client programs.

=head1 SYNOPSIS

    bi simulate ...

=head1 INHERITS

L<Bi::Client>

=head1 OPTIONS

=over 4

=item * C<-T> (default 0.0)

Length of time.

=item * C<-P> (default 1)

Number of trajectories.

=item * C<-K> (default 1)

Number of times at which to output. If 1, outputs the state at the end time
only. If greater than 1, outputs the initial state, and C<K - 1> states
equispaced in time, up to and including the end time.

=item * C<--init-file>

File from which to initialise parameters and initial conditions.

=item * C<--input-file>

File from which to read forcings.

=item * C<--output-file>

File to which to write output.

=item * C<--init-ns> (default 0)

Index along the C<ns> dimension of C<--init-file> to use.

=item * C<--input-ns> (default 0)

Index along the C<ns> dimension of C<--input-file> to use.

=item * C<--seed> (default 0)

Pseudorandom number generator seed.

=item * C<--time> (default 0)

True to time the run, false otherwise.

=item * C<--output> (default 1)

True to produce output, false otherwise.

=item * C<--threads I<N>> (default 0)

Run with C<I<N>> threads. If zero, the number of threads used is the default
for OpenMP on the platform.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'T',
      type => 'float',
      default => 0.0
    },
    {
      name => 'P',
      type => 'int',
      default => 1
    },
    {
      name => 'K',
      type => 'int',
      default => 1
    },
    {
      name => 'init-file',
      type => 'string'
    },
    {
      name => 'input-file',
      type => 'string'
    },
    {
      name => 'output-file',
      type => 'string'
    },
    {
      name => 'init-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'input-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'seed',
      type => 'int',
      default => 0
    },
    {
      name => 'time',
      type => 'int',
      default => 0
    },
    {
      name => 'output',
      type => 'int',
      default => 1
    },
    {
      name => 'threads',
      type => 'int',
      default => 0
    }
);

=head1 METHODS

=over 4

=cut

package Bi::Client::simulate;

use base 'Bi::Client';
use warnings;
use strict;

sub init {
    my $self = shift;

    $self->{_binary} = 'simulate';
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
