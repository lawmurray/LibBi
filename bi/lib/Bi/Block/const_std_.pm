=head1 NAME

const_std_ - optimisation block for L<std_> actions with arguments
that are constant expressions.

=cut

package Bi::Block::const_std_;

use parent 'Bi::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if (@{$self->get_blocks} > 0) {
        die("a 'const_std_' block may not contain nested blocks\n");
    }
    if (@{$self->get_actions} != 1) {
        die("a 'const_std_' block may only contain one action\n");
    }

    my $action = $self->get_action;
    if ($action->get_name ne 'std_') {
        die("a 'const_std_' block may only contain 'std_' actions\n");
    } else {
        foreach my $arg ('std') {
            if ($action->is_named_arg($arg) && !$action->get_named_arg($arg)->is_const) {
              die("a 'const_std_' block may only contain actions with constant '$arg' argument\n");
            }
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
