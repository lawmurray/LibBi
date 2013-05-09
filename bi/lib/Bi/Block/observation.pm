=head1 NAME

observation - the likelihood function.

=head1 SYNOPSIS

    sub observation {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<observation> block may only refer to variables of type
C<param>, C<input> and C<state>. They may only target variables of type
C<obs>.

=cut

package Bi::Block::observation;

use parent 'Bi::Block';
use warnings;
use strict;

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
        
        if ($op ne '~') {
            warn("only '~' actions should appear in the '$name' block.\n");
        } elsif ($type ne 'obs') {
            warn("variable '$var_name' is of type '$type'; only variables of type 'obs' should appear on the left side of actions in the '$name' block.\n");
        }
    }
    
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
