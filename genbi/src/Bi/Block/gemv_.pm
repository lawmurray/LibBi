=head1 NAME

gemv_ - optimisation block for L<gemv> actions.

=cut

package Bi::Block::gemv_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'gemv') {
            die("a 'gemv_' block may only contain 'gemv' actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
