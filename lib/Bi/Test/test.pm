=head1 NAME

test - frontend to test client programs.

=head1 SYNOPSIS

    libbi test ...

=head1 INHERITS

L<Bi::Client>

=cut
our @CLIENT_OPTIONS = ();

=head1 METHODS

=over 4

=cut

package Bi::Client::test;

use parent 'Bi::Client';
use warnings;
use strict;

sub init {
    my $self = shift;

    $self->{_binary} = 'test';
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
