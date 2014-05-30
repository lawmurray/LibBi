=head1 NAME

transpose - transpose a matrix.

=head1 SYNOPSIS

    B <- transpose(A)

=head1 DESCRIPTION

A C<transpose> action performs a matrix transpose.

=cut

package Bi::Action::transpose;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<A> (position 0, mandatory)

The matrix.

=back

=cut
our $ACTION_ARGS = [
{
    name => 'A',
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
    
    my $A = $self->get_named_arg('A');
    
    my $shape = [];
    push(@$shape, $A->get_shape->get_size2);
    if ($A->get_shape->get_size1 > 1) {
        push(@$shape, $A->get_shape->get_size1);
    }
    $self->set_shape(new Bi::Expression::Shape($shape));
    unless ($self->get_left->get_shape->compat($self->get_shape)) {
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
