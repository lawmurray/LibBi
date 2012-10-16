=head1 NAME

simulate - frontend to simulation client programs.

=head1 SYNOPSIS

    bi simulate ...

=head1 INHERITS

L<Bi::Client>

=head1 OPTIONS

=over 4

=item * C<--start-time> (default 0.0)

Start time.

=item * C<-T> (default 0.0)

Length of time.

=item * C<-P> (default 1)

Number of trajectories.

=item * C<-K> (default 0)

Number of dense output times. The state is always output at time C<-T> and at
all observation times in C<--obs-file>. This argument gives the number of
additional, equispaced times at which to output. For each C<k> in <0,...,K-1>,
the state will be output at time C<T*k/K>.

=item * C<--init-file>

File from which to initialise parameters and initial conditions.

=item * C<--input-file>

File from which to read inputs.

=item * C<--obs-file> (mandatory)

File from which to read observations.

=item * C<--output-file>

File to which to write output.

=item * C<--init-ns> (default 0)

Index along the C<ns> dimension of C<--init-file> to use.

=item * C<--init-np> (default 0)

Index along the C<np> dimension of C<--init-file> to use.

=item * C<--input-ns> (default 0)

Index along the C<ns> dimension of C<--input-file> to use.

=item * C<--input-np> (default 0)

Index along the C<np> dimension of C<--input-file> to use.

=item * C<--obs-ns> (default 0)

Index along the C<ns> dimension of C<--obs-file> to use.

=item * C<--obs-np> (default 0)

Index along the C<np> dimension of C<--obs-file> to use.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'start-time',
      type => 'float',
      default => 0.0
    },
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
      default => 0
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
      name => 'obs-file',
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
      name => 'init-np',
      type => 'int',
      default => 0
    },
    {
      name => 'input-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'input-np',
      type => 'int',
      default => 0
    },
    {
      name => 'obs-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'obs-np',
      type => 'int',
      default => -1
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
