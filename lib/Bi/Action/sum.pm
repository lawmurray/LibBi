=head1 NAME

sum - Sum over the elements of a vector or matrix

=head1 SYNOPSIS

    a <- sum(x)

=head1 DESCRIPTION

A C<sum> calculates the sum over all elements of an array.

=cut

package Bi::Action::sum;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<x> (position 0, mandatory)

A vector, matrix or higher-dimensional array.

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

    my $x = $self->get_named_arg('x');

    $self->set_parent('eval_');
    $self->set_can_combine(1);
    $self->set_can_nest(1);
    $self->set_unroll_target(1);
}

1;

=head1 AUTHOR

Sebastian Funk <sebastian.funk@lshtm.ac.uk>
