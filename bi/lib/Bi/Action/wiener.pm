=head1 NAME

wiener - wiener process increment action.

=head1 SYNOPSIS

    dW ~ wiener()

=head1 DESCRIPTION

A C<wiener> action specifies that a dynamic variable is distributed according
to a Wiener process.

A C<wiener> action may only be used within the L<transition> block.

=cut

package Bi::Action::wiener;

use base 'Bi::Model::Action';
use warnings;
use strict;

our $ACTION_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->set_parent('wiener_');
}

sub mean {
    # mean is always zero
    return Bi::Expression::Literal(0.0);
}

sub jacobian {
    # always zero
    return ([], []);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
