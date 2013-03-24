=head1 NAME

common_gemv_ - optimisation block for L<gemv> actions when matrix is common.

=cut

package Bi::Block::common_gemv_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'common_gemv_' block may not contain nested blocks\n");
    }
    if ($self->num_actions != 1) {
        die("a 'common_gemv_' block may only contain one action\n");
    }
    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'gemv_') {
            die("a 'common_gemv_' block may only contain 'gemv_' actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
