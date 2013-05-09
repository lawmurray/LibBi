=head1 NAME

std_ - optimisation block for L<std_> actions.

=cut

package Bi::Block::std_;

use parent 'Bi::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if (@{$self->get_blocks} > 0) {
        die("a 'std_' block may not contain nested blocks\n");
    }
    if (@{$self->get_actions} != 1) {
        die("a 'std_' block may only contain one action\n");
    }

    my $action = $self->get_action;
    if ($action->get_name ne 'std_') {
        die("a 'std_' block may only contain 'std_' actions\n");
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
