=head1 NAME

uninformative_ - block for L<uninformative> actions.

=cut

package Bi::Block::uninformative_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
       
    if ($self->num_blocks > 0) {
        die("an 'uninformative_' block may not contain sub-blocks\n");
    }
    foreach my $action (@{$self->get_actions}) {
        if ($action->get_name ne 'uninformative') {
            die("an 'uninformative_' block may only contain 'uninformative' actions\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
