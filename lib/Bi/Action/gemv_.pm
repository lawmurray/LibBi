=head1 NAME

gemv_ - dense matrix-vector multiplication.

=head1 SYNOPSIS

    y <- A*x
    y <- gemv_(A, x)
    y <- gemv_(A = A, x = x)

=head1 DESCRIPTION

A C<gemv_> action performs a dense matrix-vector multiplication. It need not
be used explicitly: any sub-expression containing the C<*> operator between a
matrix and vector is evaluated using C<gemv_>.

=cut

package Bi::Action::gemv_;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<A> (position 0, mandatory)

The matrix.

=item C<x> (position 1, mandatory)

The vector.

=back

=cut
our $ACTION_ARGS = [
{
    name => 'A',
    positional => 1,
    mandatory => 1
  },
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
    $self->ensure_matrix('A');
    $self->ensure_vector('x');

    my $A = $self->get_named_arg('A');
    my $x = $self->get_named_arg('x');

    if ($A->get_shape->[1] != $x->get_shape->[0]) {
        die("incompatible dimension sizes in arguments to action 'gemv_'");
    } else {
        $self->set_shape([ $A->get_shape->[0] ]);
    }

    if ($A->is_common) {
        $self->set_parent('common_gemv_');
        $self->set_can_combine(0);
    } else {
        $self->set_parent('matrix_');
        $self->set_can_combine(1);
    }
    $self->set_is_matrix(1);
    $self->set_can_nest(1);
    $self->set_unroll_target(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
