=head1 NAME

transition - the transition distribution.

=head1 SYNOPSIS

    sub transition(delta = 1.0) {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<transition> block may reference variables of any
type except C<obs>, but may only target variables of type C<noise> and
C<state>.

=cut

package Bi::Block::transition;

use parent 'Bi::Block';
use warnings;
use strict;

use Bi::Utility qw(contains);

=head1 PARAMETERS

=over 4

=item C<delta> (position 0, default 1.0)

The time step for discrete-time components of the transition. Must be a
constant expression.

=back

=cut

our $BLOCK_ARGS = [
  {
    name => 'delta',
    positional => 1,
    default => 1.0
  }
];

sub validate {
    my $self = shift;
    
    my $name = $self->get_name;
    $self->process_args($BLOCK_ARGS);
    if (!$self->get_named_arg('delta')->is_const) {
        die("argument 'delta' to block '$name' must be a constant expression\n");
    }
    
    my $actions = $self->get_all_actions;
    foreach my $action (@$actions) {
        my $op = $action->get_op;
        my $var_name = $action->get_left->get_var->get_name;
        my $type = $action->get_left->get_var->get_type;
        
        if ($op eq '~' && $type ne 'noise') {
            warn("variable '$var_name' is of type '$type'; only 'noise' variables should appear on the left side of '~' actions in the '$name' block.\n");
        } elsif ($op eq '<-' && !contains(['state', 'state_aux_'], $type)) {
            warn("variable '$var_name' is of type '$type'; only 'state' variables should appear on the left side of '<-' actions in the '$name' block.\n");
        } elsif ($op eq '=' && !contains(['state', 'state_aux_'], $type)) {
            warn("variable '$var_name' is of type '$type'; only 'state' variables should appear on the left side of '=' actions in the '$name' block.\n");
        }
        
        my $refs = $action->get_right_var_refs;
        foreach my $ref (@$refs) {
            if ($ref->get_var->get_type eq 'obs') {
                warn("'obs' variables should not appear on the right side of actions in the '$name' block.\n");
            }
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
