=head1 NAME

binomial - Binomial distribution.

=head1 SYNOPSIS

    x ~ binomial()
    x ~ binomial(1, 0.5)
    x ~ binomial(size = 1, prob = 0.5)

=head1 DESCRIPTION

A C<binomial> action specifies that a variable is distributed according to
a binomial distribution with the given C<size> and C<prob> parameters.
Note that the implementation will evaluate densities for any (not necessarily
integer) x. It is left to the user to ensure consistency (e.g., using this only
with integer observations).


=cut

package Bi::Action::binomial;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<size> (position 0, default 1)

Mean parameter of the distribution.

=item C<prob> (position 1, default 0.5)

Shape parameter of the distribution.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'size',
    positional => 1,
    default => 1
  },
  {
    name => 'prob',
    positional => 1,
    default => 0.5
  },
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('size');
    $self->ensure_scalar('prob');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    # size*prob
    my $size = $self->get_named_arg('size')->clone;
    my $prob = $self->get_named_arg('prob')->clone;
    my $mean = $size*$prob;

    return $mean;
}

1;

=head1 AUTHOR

Edwin van Leeuwen <edwinvanl@gmail.com>

=head1 VERSION

$Rev$ $Date$
