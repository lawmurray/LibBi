=head1 NAME

parameter - the prior distribution over parameters.

=head1 SYNOPSIS

    sub parameter {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<parameter> block may only refer to variables of type C<input>
and C<param>. They may only target variables of type C<param>.

=cut

package Bi::Block::parameter;

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
        
        if (!contains(['param', 'param_aux_'], $type)) {
            warn("variable '$var_name' is of type '$type'; only variables of type 'param' should appear on the left side of actions in the '$name' block.\n");
        }
        
        my $refs = $action->get_right_var_refs;
        foreach my $ref (@$refs) {
            my $var_name = $ref->get_var->get_name;
            my $type = $ref->get_var->get_type;
            
            if (!contains(['input', 'param', 'param_aux_'], $type)) {
                warn("variable '$var_name' is of type '$type'; only variables of type 'input' and 'param' should appear on the right side of actions in the '$name' block.\n");
            }
        }
    }
    
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
