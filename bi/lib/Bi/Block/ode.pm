=head1 NAME

ode - block for control of ordinary differential equation actions.

=head1 SYNOPSIS

    ode(alg = 'rk43', h = 1.0, atoler = 1.0e-3, rtoler = 1.0e-3) {
      x1 <- ode(...)
      x2 <- ode(...)
      ...
    }

    ode('rk43', 1.0, 1.0e-3, 1.0e-3) {
      x1 <- ode(...)
      x2 <- ode(...)
      ...
    }

=head1 DESCRIPTION

An C<ode> block is used to group multiple L<ode> actions and configure the
numerical integrator used to apply them.

An ode block may only be used within a L<transition> block, may not contain
nested blocks, and may only contain L<ode> actions.

=cut

package Bi::Block::ode;

use base 'Bi::Model::Block';
use warnings;
use strict;

use Carp::Assert;

use Bi::Jacobian;
use Bi::Utility qw(find);

=head1 PARAMETERS

=over 4

=item C<alg> (position 0, default C<'rk43'>)

The numerical integration algorithm to be used. Valid values are:

=over 8

=item C<'rk4'>

The classic order 4 Runge-Kutta with fixed step size.

=item C<'dopri5'>

An order 5(4) Dormand-Prince with adaptive step size control.

=item C<'rk43'>

An order 4(3) low-storage Runge-Kutta with adaptive step size control.

=back

=item C<h> (position 1, default 1.0)

For a fixed step size, the step size to use. For an adaptive step size, the
suggested initial step size to use.

=item C<atoler> (position 2, default 1.0e-3)

The absolute error tolerance for adaptive step size control.

=item C<rtoler> (position 3, default 1.0e-3)

The relative error tolerance for adaptive step size control.

=back

=cut
our $BLOCK_ARGS = [
  {
    name => 'alg',
    positional => 1,
    default => "'rk43'"
  },
  {
    name => 'h',
    positional => 1,
    default => 1.0
  },
  {
    name => 'atoler',
    positional => 1,
    default => 0.001
  },
  {
    name => 'rtoler',
    positional => 1,
    default => 0.001
  }
];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    my $alg = $self->get_named_arg('alg')->eval_const;
    if ($alg ne 'rk4' && $alg ne 'dopri5' && $alg ne 'rk43') {
        die("unrecognised value '$alg' for argument 'alg' of block 'ode'\n");
    }
}

sub add_extended_actions {
    my $self = shift;
    my $model = shift;
    my $vars = shift;
    my $J_commit = shift;
    my $J_new = shift;
    
    # get and then clear actions, will be replaced
    my $actions = $self->get_actions;
    $self->set_actions([]);
    
    # inplace operations, so need to add all variables in new Jacobian to
    # existing Jacobian
    my %ids;
    foreach my $action (@$actions) {
        my ($ds, $refs) = $action->jacobian;
    	my $i = find($vars, $action->get_left->get_var);
    	$ids{$i} = 1;
    	foreach my $ref (@$refs) {
    		$i = find($vars, $ref->get_var);
    		$ids{$i} = 1;
    	}
    }
    foreach my $i (keys %ids) {
        foreach my $j (keys %ids) {
        	if ($vars->[$j]->get_type ne 'noise') {
		        my $J_var = $model->get_jacobian_var($vars->[$i], $vars->[$j]); 
		        my $ref = new Bi::Expression::VarIdentifier($J_var);
	            $J_commit->set($i, $j, $ref);
        	}
        }
    }
    
    foreach my $action (@$actions) {
        # search for index that corresponds to the target of this action
        my $j = find($vars, $action->get_left->get_var);
        assert ($j >= 0) if DEBUG;
        
        # mean
        my $mean = $action->mean;
        if (defined $mean) {
            my $id = $model->next_action_id;
            my $target = $action->get_target->clone;
            my $action = new Bi::Model::Action($id, $target, '<-', $mean);
            
            $self->push_action($action);
        }

        # Jacobian
        my ($ds, $refs) = $action->jacobian;

        for (my $l = 0; $l < $J_commit->num_rows; ++$l) {
        	my $expr;
	        for (my $k = 0; $k < @$ds; ++$k) {
	            my $ref = $refs->[$k];
	            my $d = $ds->[$k];
	            my $i = find($vars, $ref->get_var);
	            if ($i >= 0) {
	                $J_new->set($i, $j, $d->clone);
                
                	if (defined $J_commit->get($l, $i)) {
                        my $arg = $J_commit->get($l, $i)->clone;
		                my @indexes = map { $_->clone } (@{$vars->[$l]->gen_indexes}, @{$ref->get_indexes});
                        $arg->set_indexes(\@indexes);

						if (!defined $expr) {
							$expr = $d->clone*$arg;
						} else {
	                        $expr += $d->clone*$arg;
						}
                	}
                }
            }
            if (defined $expr) {
            	my $id = $model->next_action_id;
		        my $var = $model->get_jacobian_var($vars->[$l], $vars->[$j]);
		        my @aliases = map { $_->clone } (@{$vars->[$l]->gen_aliases}, @{$action->get_target->get_aliases});
		        my $target = new Bi::Model::Target($var, \@aliases);
	            my $action = new Bi::Model::Action($id, $target, '<-', new Bi::Expression::Function('ode', [ $expr ]));
	            $self->push_action($action);
            }
        }
    }
    #_inline($model, $J);
    #$self->add_mean_actions($model, $vars, $mu);
    #$self->add_std_actions($model, $vars, $S);
    #$self->add_jacobian_actions($model, $vars, $J_commit, $J);

    if ($self->get_commit) {
        $J_commit->swap(Bi::Jacobian::commit($model, $vars, $J_commit*$J_new));
        $J_new->ident;
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
