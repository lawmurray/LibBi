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
C<sample> command when the C<--transform-initial-to-param> option is
used.

Actions in the C<proposal_initial> block may only refer to variables of
type C<param>, C<input> and C<state>. They may only target variables of type
C<state>.

=cut

package Bi::Block::proposal_initial;

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
