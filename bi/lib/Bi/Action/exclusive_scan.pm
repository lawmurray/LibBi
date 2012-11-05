=head1 NAME

exclusive_scan - exclusive scan (prefix sum, cumulative sum).

=head1 SYNOPSIS

    X <- exclusive_scan(x)

=head1 DESCRIPTION

An C<exclusive_scan> action computes into each element C<i> of C<X>, the sum
of the first C<i - 1> elements of C<x>.

=cut

package Bi::Action::exclusive_scan;

use base 'Bi::Model::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item * C<x> (position 0, mandatory)

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
    
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('<-');
    $self->ensure_vector('x');

    my $x = $self->get_named_arg('x');
    $self->set_dims([ $x->get_dims->[0] ]);

    $self->set_parent('eval');
    $self->set_is_matrix(1);
    $self->set_can_nest(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
