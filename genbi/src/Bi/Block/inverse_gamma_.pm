=head1 NAME

inverse_gamma_ - block for L<inverse_gamma> actions.

=cut

package Bi::Block::inverse_gamma_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'inverse_gamma_' block may not contain sub-blocks\n");
    }
    if ($self->num_actions != 1) {
        die("a 'inverse_gamma_' block may only contain one action\n");
    }

    my $action = $self->get_action;
    if ($action->get_name ne 'inverse_gamma') {
        die("a 'inverse_gamma_' block may only contain 'inverse_gamma' actions\n");
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
