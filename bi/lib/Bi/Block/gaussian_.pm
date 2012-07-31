=head1 NAME

gaussian_ - block for L<gaussian> actions.

=cut

package Bi::Block::gaussian_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'gaussian_' block may not contain sub-blocks\n");
    }

    my $action = $self->get_action;
    if ($action->get_name ne 'gaussian' && $action->get_name ne 'normal' && $action->get_name ne 'log_gaussian' && $action->get_name ne 'log_normal') {
        die("a 'gaussian_' block may only contain 'gaussian', 'normal', 'log_gaussian' and 'log_normal' actions\n");
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
