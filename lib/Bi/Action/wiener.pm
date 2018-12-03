=head1 NAME

wiener - wiener process.

=head1 SYNOPSIS

    dW ~ wiener()

=head1 DESCRIPTION

A C<wiener> action specifies that a variable is an increment of a Wiener
process: Gaussian distributed with mean zero and variance C<tj - ti>,
where C<ti> is the starting time, and C<tj> the ending time, of the current
time interval

=cut

package Bi::Action::wiener;

use parent 'Bi::Action';
use warnings;
use strict;

our $ACTION_ARGS = [];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('wiener_');
}

sub mean {
    # mean is always zero
    return new Bi::Expression::Literal(0.0);
}

sub jacobian {
    # always zero
    return ([], []);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

