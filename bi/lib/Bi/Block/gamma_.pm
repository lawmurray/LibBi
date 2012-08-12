=head1 NAME

gamma_ - block for L<gamma> actions.

=cut

package Bi::Block::gamma_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'gamma_' block may not contain sub-blocks\n");
    }
    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'gamma') {
            die("a 'gamma_' block may only contain 'gamma' actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
