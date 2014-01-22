=head1 NAME

inclusive_scan - inclusive scan primitive (also called prefix sum or
cumulative sum).

=head1 SYNOPSIS

    X <- inclusive_scan(x)

=head1 DESCRIPTION

An C<inclusive_scan> action computes into each element C<i> of C<X>, the sum
of the first C<i> elements of C<x>.

=cut

package Bi::Action::inclusive_scan;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<x> (position 0, mandatory)

The vector over which to scan.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'x',
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

    my $x = $self->get_named_arg('x');
    $self->set_shape($x->get_shape);

    unless ($self->get_left->get_shape->equals($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('matrix_');
    $self->set_is_matrix(1);
    $self->set_can_nest(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
