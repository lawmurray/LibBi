=head1 NAME

gemm_ - dense matrix-matrix multiplication.

=head1 SYNOPSIS

    Y <- A*X
    Y <- gemm_(A, X)
    Y <- gemm_(A = A, X = X)

=head1 DESCRIPTION

A C<gemm_> action performs a dense matrix-matrix multiplication. It need not
be used explicitly: any sub-expression containing the C<*> operator between
two matrices is evaluated using C<gemm_>.

=cut

package Bi::Action::gemm_;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<A> (position 0, mandatory)

A matrix.

=item C<X> (position 1, mandatory)

A matrix.

=back

=cut
our $ACTION_ARGS = [
{
    name => 'A',
    positional => 1,
    mandatory => 1
  },
  {
    name => 'X',
    positional => 1,
    mandatory => 1
  }  
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');
    $self->ensure_matrix('A');
    $self->ensure_matrix('X');

    my $A = $self->get_named_arg('A');
    my $X = $self->get_named_arg('X');

    if ($A->get_shape->get_sizes->[1] != $X->get_shape->get_sizes->[0]) {
        die("incompatible dimension sizes in arguments to action 'gemm_'");
    } else {
        $self->set_shape(new Bi::Expression::Shape([ $A->get_shape->get_sizes->[0], $X->get_shape->get_sizes->[1] ]));
    }
    unless ($self->get_left->get_shape->equals($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('matrix_');
    $self->set_can_combine(1);
    $self->set_is_matrix(1);
    $self->set_can_nest(1);
    $self->set_unroll_target(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>
