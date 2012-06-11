=head1 NAME

sample - frontend to sampling client programs.

=head1 SYNOPSIS

    bi sample ...
    
=head1 INHERITS

L<Bi::Client::filter>

=cut

package Bi::Client::sample;

use base 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<sample> program inherits all options from C<filter>, and permits the
following additional options:

=over 4

=item * C<-D> (default 1)

Number of samples to draw.

=item * C<--filter-file>

File from which to read and write intermediate filter results.

=item * C<--include-initial> (default 0)

Include initial conditions in outer Metropolis-Hastings loop (as opposed to
inner filtering loop)?

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'D',
      type => 'int',
      default => 1
    },
    {
      name => 'filter-file',
      type => 'string'
    },
    {
      name => 'include-initial',
      type => 'int',
      default => 0
    }
);

sub init {
    my $self = shift;

    Bi::Client::filter::init($self);
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
    my $self = shift;

    $self->Bi::Client::filter::process_args(@_);
    $self->{_binary} = 'pmmh';
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
