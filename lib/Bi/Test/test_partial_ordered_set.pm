=head1 NAME

test_partial_ordered_set - test partial_ordered_set class.

=head1 SYNOPSIS

    libbi test_partial_ordered_set ...

=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Test::test_partial_ordered_set;

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

    $self->{_binary} = 'test_partial_ordered_set';
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
