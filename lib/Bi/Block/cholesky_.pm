=head1 NAME

cholesky_ - special block for cholesky action.

=cut

package Bi::Block::cholesky_;

use parent 'Bi::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    my $name = $self->get_name;
    if (@{$self->get_blocks} > 0) {
        die("a '$name' block may not contain nested blocks\n");
    }
    if (@{$self->get_actions} != 1) {
        die("a '$name' block may only contain one action\n");
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>
