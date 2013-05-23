=head1 NAME

initial - the prior distribution over the initial values of state variables.

=head1 SYNOPSIS

    sub initial {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<initial> block may only refer to variables of type
C<param>, C<input> and C<state>. They may only target variables of type
C<state>.

=cut

package Bi::Block::initial;

use parent 'Bi::Block';
use warnings;
use strict;

use Bi::Utility qw(contains);

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    my $name = $self->get_name;
    my $actions = $self->get_all_actions;
    foreach my $action (@$actions) {
        my $op = $action->get_op;
        my $var_name = $action->get_left->get_var->get_name;
        my $type = $action->get_left->get_var->get_type;
        
        if (!contains(['state', 'state_aux_'], $type)) {
            warn("variable '$var_name' is of type '$type'; only variables of type 'state' should appear on the left side of actions in the '$name' block.\n");
        }
        
        my $refs = $action->get_right_var_refs;
        foreach my $ref (@$refs) {
            my $var_name = $ref->get_var->get_name;
            my $type = $ref->get_var->get_type;
            
            if (!contains(['input', 'param', 'param_aux_', 'state'], $type)) {
                warn("variable '$var_name' is of type '$type'; only variables of type 'input', 'param' and 'state' should appear on the right side of actions in the '$name' block.\n");
            }
        }
    }    
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
