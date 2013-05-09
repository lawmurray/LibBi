=head1 NAME

lookahead_transition - a transition distribution for lookahead operations.

=head1 SYNOPSIS

    sub lookahead_transition {
      ...
    }
    
=head1 DESCRIPTION

This may be a deterministic, computationally cheaper or perhaps inflated
version of the transition distribution. It is used by the auxiliary particle
filter.

Actions in the C<lookahead_transition> block may reference variables of any
type except C<obs>, but may only target variables of type C<noise> and
C<state>.

=back

=cut

package Bi::Block::lookahead_transition;

use parent 'Bi::Block::transition';
use warnings;
use strict;

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
