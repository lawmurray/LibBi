=head1 NAME

smooth - frontend to smoothing client programs.

=head1 SYNOPSIS

    bi smooth ...
    
=head1 INHERITS

L<Bi::Client::Simulate>

=cut

package Bi::Client::Smooth;

use base 'Bi::Client';
use warnings;
use strict;

=head1 OPTIONS

=over 4

=item C<--smoother> (default C<'pfs'>)

The type of smoother to use; one of:

=over 8

=item C<'pfs'>

for a particle filter-smoother,

=item C<'rtss'>

for a Rauch-Tung-Striebel (RTS) smoother.

=back

=item C<--filter-file> (mandatory)

File from which to read filter results.

=item C<--output-file>

File to which to write output.

=item C<--time> (default 0)

True to time the run, false otherwise.

=item C<--output> (default 1)

True to produce output, false otherwise.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'smoother',
      type => 'string',
      default => 'pfs'
    },
    {
      name => 'filter-file',
      type => 'string'
    },
    {
      name => 'output-file',
      type => 'string'
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
    }
);

sub new {
    my $self = shift;

    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
    my $self = shift;

    $self->Bi::Client::process_args(@_);
    $self->{_binary} = $self->get_named_arg('smoother');
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
