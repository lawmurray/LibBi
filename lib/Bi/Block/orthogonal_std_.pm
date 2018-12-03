=head1 NAME

orthogonal_std_ - optimisation block for L<std_> actions with
orthogonal covariance matrix.

=cut

package Bi::Block::orthogonal_std_;

use parent 'Bi::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    my ($name, $action, $arg);
    
    if (@{$self->get_blocks} > 0) {
        die("an 'orthogonal_std_' block may not contain nested blocks\n");
    }
    foreach $action (@{$self->get_actions}) {
        if ($action->get_name ne 'std_' && $action->get_name ne 'normal' && $action->get_name ne 'log_std_' && $action->get_name ne 'log_normal') {
            die("an 'orthogonal_std_' block may only contain 'std_', 'normal', 'log_std_' and 'log_normal' actions\n");         
        } else {
            foreach $name ('std') {
                if ($action->is_named_arg($name)) {
                    $arg = $action->get_named_arg($name);
                    if (!$arg->is_element && !$arg->is_scalar && !$arg->is_vector) {
                        die("an 'orthogonal_std_' block may only contain actions with element, scalar or vector '$name' argument\n");
                    }
                }
            }
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

