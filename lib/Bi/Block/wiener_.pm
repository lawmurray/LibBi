=head1 NAME

wiener_ - block for L<wiener> actions.

=cut

package Bi::Block::wiener_;

use parent 'Bi::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if (@{$self->get_blocks} > 0) {
        die("a 'wiener_' block may not contain nested blocks\n");
    }

    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'wiener') {
            die("a 'wiener_' block may only contain 'wiener' actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

