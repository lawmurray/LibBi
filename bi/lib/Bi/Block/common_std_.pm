=head1 NAME

common_std_ - optimisation block for L<std_> actions with arguments
that are common expressions.

=cut

package Bi::Block::common_std_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'common_std_' block may not contain nested blocks\n");
    }
    if ($self->num_actions != 1) {
        die("a 'common_std_' block may only contain one action\n");
    }

    my $action = $self->get_action;
    if ($action->get_name ne 'std_') {
        die("a 'common_std_' block may only contain 'std_' actions\n");
    } else {
        foreach my $arg ('std') {
            if ($action->is_named_arg($arg) && !$action->get_named_arg($arg)->is_common) {
              die("a 'common_std_' block may only contain actions with common '$arg' argument\n");
            }
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
