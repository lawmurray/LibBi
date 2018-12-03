=head1 NAME

common_orthogonal_std_ - optimisation block for L<std_> actions with
orthogonal covariance matrix and arguments that are common expressions.

=cut

package Bi::Block::common_orthogonal_std_;

use parent 'Bi::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    my ($name, $action, $arg);
    
    if (@{$self->get_blocks} > 0) {
        die("a 'common_orthogonal_std_' block may not contain nested blocks\n");
    }
    foreach $action (@{$self->get_actions}) {
        if ($action->get_name ne 'std_') {
            die("a 'common_orthogonal_std_' block may only contain 'std_' actions\n");            
        } else {
            foreach $name ('std') {
                if ($action->is_named_arg($name) && !$action->get_named_arg($name)->is_common) {
                    die("a 'common_orthogonal_std_' block may only contain actions with common '$name' argument\n");
                }
            }
            foreach $name ('std') {
                if ($action->is_named_arg($name)) {
                    $arg = $action->get_named_arg($name);
                    if (!$arg->is_scalar && !$arg->is_vector) {
                        die("a 'common_orthogonal_std_' block may only contain actions with element, scalar or vector '$name' argument\n");
                    }
                }
            }
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

