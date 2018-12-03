=head1 NAME

proposal_initial - a proposal distribution over the initial values of state
variables.

=head1 SYNOPSIS

    sub proposal_initial {
      x ~ gaussian(x, 1.0)    // local proposal
      x ~ gaussian(0.0, 1.0)  // independent proposal
    }
    
=head1 DESCRIPTION

This may be a local or independent proposal distribution, used by the
C<sample> command when the C<--with-transform-initial-to-param> option is
used.

Actions in the C<proposal_initial> block may only refer to variables of
type C<param>, C<input> and C<state>. They may only target variables of type
C<state>.

=cut

package Bi::Block::proposal_initial;

use parent 'Bi::Block::initial';
use warnings;
use strict;

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

