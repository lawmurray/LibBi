=head1 NAME

beta_ - block for L<beta> actions.

=cut

package Bi::Block::beta_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'beta_' block may not contain sub-blocks\n");
    }
    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'beta') {
            die("a 'beta_' block may only contain 'beta' actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 2925 $ $Date: 2012-08-12 17:49:28 +0800 (Sun, 12 Aug 2012) $
