=head1 NAME

cholesky - Cholesky factorisation.

=head1 SYNOPSIS

    U <- cholesky(A)
    L <- cholesky(A, 'L')

=head1 DESCRIPTION

A C<cholesky> action performs a Cholesky factorisation of a symmetric
positive definite matrix, returning either the lower- or upper-triangular
factor.

=cut

package Bi::Action::cholesky;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<A> (position 0, mandatory)

The matrix.

=item C<uplo> (position 1, default C<'U'>)

C<'U'> to return the upper-triangular factor, C<'L'> to return the
lower-triangular factor.

=back

=cut
our $ACTION_ARGS = [
{
    name => 'A',
    positional => 1,
    mandatory => 1
  },
  {
    name => 'uplo',
    positional => 1,
    default => 'U'
  }  
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');
    $self->ensure_matrix('A');

    my $A = $self->get_named_arg('A');
    my $uplo = $self->get_named_arg('uplo')->eval_const;
    if ($A->get_shape->get_sizes->[0] != $A->get_shape->get_sizes->[1]) {
    	die("argument 'A' of action 'cholesky' must be a square matrix\n");
    } else {
        $self->set_shape(new Bi::Expression::Shape([ $A->get_shape->get_sizes->[0], $A->get_shape->get_sizes->[1] ]));
    }
    if ($uplo ne 'U' && $uplo ne 'L') {
        die("unrecognised value '$uplo' for argument 'uplo' of action 'cholesky'\n");
    }

    $self->set_parent('cholesky_');
    $self->set_is_matrix(1);
    $self->set_can_nest(1);
    $self->set_unroll_target(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>
