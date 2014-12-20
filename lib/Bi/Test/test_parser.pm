=head1 NAME

test_parser - test new Flex- and Bison-based parser.

=head1 SYNOPSIS

    libbi test_parser ...

=head1 INHERITS

L<Bi::Client>

=cut

package Bi::Test::test_parser;

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

	$self->{_binary} = 'test_parser';
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
