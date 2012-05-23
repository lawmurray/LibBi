=head1 NAME

lookahead_observation - declare a lookahead density to accompany the
observation density.

=head1 SYNOPSIS

    sub lookahead_observation {
      ...
    }
    
=head1 DESCRIPTION

Use the C<lookahead_observation> block to specify a lookahead density to
accompany the observation density. This may be a deterministic,
computationally cheaper or perhaps inflated version of the observation
density, for example. It is used by methods such as the auxiliary particle
filter. 

Actions in the C<lookahead_observation> block may only refer to variables
of type C<param>, C<force> and C<state>. They may only target variables of
type C<obs>, using the C<~> operator.

=cut

package Bi::Block::lookahead_observation;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
