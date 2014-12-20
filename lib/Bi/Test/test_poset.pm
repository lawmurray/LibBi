=head1 NAME

test_poset - test poset class.

=head1 SYNOPSIS

    libbi test_poset ...

=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Test::test_poset;

use parent 'Bi::Client';
use warnings;
use strict;

=head1 OPTIONS

=over 4

=back

=cut
our @CLIENT_OPTIONS = ();

sub init {
    my $self = shift;

    $self->{_binary} = 'test_poset';
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub needs_model {
    return 0;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
