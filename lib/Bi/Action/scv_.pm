=head1 NAME

scv_ - scalar vector-vector multiplication

=head1 SYNOPSIS

    a <- x.y
    a <- scv_(x, y)
    a <- scv_(x = x, y = y)

=head1 DESCRIPTION

A C<scv_> action performs a scalar vector-vector (dot) multiplication. It need not
be used explicitly: any sub-expression containing the C<.> operator between two
vectors is evaluated using C<scv_>.

=cut

package Bi::Action::scv_;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<x> (position 0, mandatory)

The first vector.

=item C<y> (position 1, mandatory)

The second vector.

=back

=cut
our $ACTION_ARGS = [
{
    name => 'x',
    positional => 1,
    mandatory => 1
  },
  {
    name => 'y',
    positional => 1,
    mandatory => 1
  }  
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');
    $self->ensure_vector('x');
    $self->ensure_vector('y');

    my $x = $self->get_named_arg('x');
    my $y = $self->get_named_arg('y');

    if ($x->get_shape->get_sizes->[0] != $y->get_shape->get_sizes->[0]) {
        die("incompatible dimension sizes in arguments to action 'scv_'");
    } else {
        $self->set_shape(new Bi::Expression::Shape());
    }
    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('eval_');
    $self->set_can_combine(1);
    $self->set_can_nest(1);
    $self->set_unroll_target(1);
}

1;

=head1 AUTHOR

Sebastian Funk <sebastian.funk@lshtm.ac.uk>
