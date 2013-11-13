=head1 NAME

bridge - the bridge potential.

=head1 SYNOPSIS

    sub bridge(delta = 1.0) {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<bridge> block may reference variables of any type, but may
only target variables of type C<noise> and C<state>. References to C<obs>
variables provide their next value. Use of the built-in variables C<t_now>
and C<t_next_obs> will be useful.

=cut

package Bi::Block::bridge;

use parent 'Bi::Block';
use warnings;
use strict;

use Bi::Utility qw(contains);

=head1 PARAMETERS

=over 4

=item C<delta> (position 0, default 1.0)

The time step for bridge weighting. Must be a constant expression.

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
        
        if ($op ne '~') {
            warn("only '~' actions should appear in the '$name' block.\n");
        } elsif ($type ne 'state' && $type ne 'noise') {
            warn("variable '$var_name' is of type '$type'; only variables of type 'state' or 'noise' should appear on the left side of actions in the '$name' block.\n");
        }
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
