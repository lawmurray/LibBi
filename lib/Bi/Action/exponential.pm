=head1 NAME

exponential - Exponential distribution.

=head1 SYNOPSIS

    x ~ exponential()
    x ~ exponential(1)
    x ~ exponential(lambda = 1)

=head1 DESCRIPTION

A C<exponential> action specifies that a variable is distributed according to
an exponential distribution with the given rate C<lambda>

=cut

package Bi::Action::exponential;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<lambda> (position 0, default 1)

Rate

=back

=cut
our $ACTION_ARGS = [
    {
        name => 'lambda',
        positional => 1,
        default => 1.0
    }
];

sub validate {
    my $self = shift;

    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('lambda');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    my $lambda = $self->get_named_arg('lambda')->clone;
    my $mean = 1/$lambda;

    return $mean;
}

1;

=head1 AUTHOR

Sebastian Funk <sebastian.funk@lshtm.ac.uk>

=head1 VERSION

$Rev$ $Date$
