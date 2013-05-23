=head1 NAME

lookahead_observation - a likelihood function for lookahead operations.

=head1 SYNOPSIS

    sub lookahead_observation {
      ...
    }
    
=head1 DESCRIPTION

This may be a deterministic, computationally cheaper or perhaps inflated
version of the likelihood function. It is used by the auxiliary particle
filter. 

Actions in the C<lookahead_observation> block may only refer to variables
of type C<param>, C<input> and C<state>. They may only target variables of
type C<obs>.

=cut

package Bi::Block::lookahead_observation;

use parent 'Bi::Block::observation';
use warnings;
use strict;

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
