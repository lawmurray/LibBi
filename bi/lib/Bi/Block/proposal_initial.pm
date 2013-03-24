=head1 NAME

proposal_initial - declare the proposal density over initial conditions.

=head1 SYNOPSIS

    sub proposal_initial {
      ...
    }
    
=head1 DESCRIPTION

Use the C<proposal_initial> block to specify the proposal density over the
initial conditions of the model.

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
