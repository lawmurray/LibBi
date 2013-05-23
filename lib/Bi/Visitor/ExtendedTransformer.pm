=head1 NAME

Bi::Visitor::ExtendedTransformer - visitor for constructing linearised
version of a block.

=head1 SYNOPSIS

    use Bi::Visitor::ExtendedTransformer;
    Bi::Visitor::ExtendedTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::ExtendedTransformer;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

use Bi::Utility qw(find);
use Bi::Expression::Matrix;

=item B<evaluate>(I<model>)

Construct and evaluate.

=over 4

=item I<model>

L<Bi::Model> object.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $r_vars = $model->get_all_vars('noise');
    my $d_vars = $model->get_all_vars('state');
    my $o_vars = $model->get_all_vars('obs');
    my $vars = [ @$r_vars, @$d_vars, @$o_vars ];

    my $NR = @$r_vars;
    my $ND = @$d_vars;
    my $NO = @$o_vars;
    my $N = @$vars;

    my $self = {
        _NR => $NR,
        _ND => $ND,
        _NO => $NO
    };
    bless $self, $class;
        
    my $J_vars = new Bi::Expression::Matrix($N, $N);
    my $F_vars = $J_vars->subrange(0, $NR + $ND, 0, $NR + $ND);
    my $G_vars = $J_vars->subrange(0, $NR + $ND, $NR + $ND, $NO);

    $F_vars->assign(_add_F_vars($model));
    $G_vars->assign(_add_G_vars($model));

    my $S_vars = new Bi::Expression::Matrix($N, $N);
    my $Q_vars = $S_vars->subrange(0, $NR + $ND, 0, $NR + $ND);
    my $R_vars = $S_vars->subrange($NR + $ND, $NO, $NR + $ND, $NO);

    $Q_vars->assign(_add_Q_vars($model));
    $R_vars->assign(_add_R_vars($model));

    my $J = new Bi::Expression::Matrix($N, $N);

    $J->subrange(0, $NR + $ND, 0, $NR + $ND)->ident;
    $model->get_block('initial')->accept($self, $model, $J, $vars, $J_vars, $S_vars);
    
    $J->clear;
    $J->subrange(0, $NR + $ND, 0, $NR + $ND)->ident;
    $model->get_block('transition')->accept($self, $model, $J, $vars, $J_vars, $S_vars);
    
    $J->clear;
    $J->subrange(0, $NR + $ND, 0, $NR + $ND)->ident;
    $model->get_block('observation')->accept($self, $model, $J, $vars, $J_vars, $S_vars);
}

=item B<visit_after>(I<node>, I<J>)

Visit node of model

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $model = shift;
    my $J = shift;
    my $vars = shift;    
    my $J_vars = shift;
    my $S_vars = shift;

    my $N = @$vars;
    
    assert ($J->size1 == $N) if DEBUG;
    assert ($J->size2 == $N) if DEBUG;
    assert ($J_vars->size1 == $N) if DEBUG;
    assert ($J_vars->size2 == $N) if DEBUG;
    assert ($S_vars->size1 == $N) if DEBUG;
    assert ($S_vars->size2 == $N) if DEBUG;
    
    my @results;
    
    if ($node->isa('Bi::Action') && !$node->is_inplace) {
     	push(@results, $self->_create_jacobian_actions($node, $J, $vars, $J_vars));
   	    push(@results, $self->_create_std_action($node, $vars, $S_vars));
        push(@results, $self->_create_mean_action($node));
    } elsif ($node->isa('Bi::Block')) {
        # inplace actions
        my @actions = map { ($_->is_inplace) ? $_ : () } @{$node->get_actions};
        
        
        # fill $J with all pairs of left variables, as these will be nonzero
        my @left_vars;
        foreach my $action (@actions) {
            push(@left_vars, $action->get_left->get_var);
        }
        foreach my $var1 (@left_vars) {
            my $i = find($vars, $var1);
            foreach my $var2 (@left_vars) {
                my $j = find($vars, $var2);
                $J->set($i, $j, $J_vars->get($i, $j));
            }
        }

        # change children
        my $children = $node->get_children;
        $node->set_children([]);
        foreach my $child (@$children) {
        	if ($child->isa('Bi::Action') && $child->is_inplace) {
        		# inplace action
        	    $node->push_children([ $self->_create_jacobian_actions($child, $J, $vars, $J_vars) ]);
	        	$node->push_children([ $self->_create_std_action($child, $vars, $S_vars) ]);
    	    	$node->push_children([ $self->_create_mean_action($child) ]);
        	} else {
        		# other block or action
        		$node->push_child($child);
        	}
        }
        push(@results, $node);
    } else {
        push(@results, $node);
    }
    
    return @results[0..$#results];
}

=item B<_create_mean_action>(I<node>)

=cut
sub _create_mean_action {
    my $self= shift;
    my $node = shift;
    
    my $mean = $node->mean;
    my $left = $node->get_left->clone;
    my $right = $mean;
        
    my $action = new Bi::Action;
    $action->set_aliases($node->get_aliases);
    $action->set_left($left);
    $action->set_op('<-');
    $action->set_right($right);
    $action->validate;
        
    return $action;
}

=item B<_create_std_action>(I<node>, I<vars>, I<S_vars>)

=cut
sub _create_std_action {
    my $self= shift;
    my $node = shift;
    my $vars = shift;
    my $S_vars = shift;
    
    my $std = $node->std;
	my $j = find($vars, $node->get_left->get_var);
    my $left = $S_vars->get($j, $j);
    my $right = $std;
    
    if (defined $std) {
        my $ranges = $left->get_indexes;
        my @aliases = map { new Bi::Model::DimAlias(undef, $_->clone) } @$ranges;
        
        my $action = new Bi::Action;
        $action->set_aliases(\@aliases);
        $action->set_left($left);
        $action->set_op('<-');
        $action->set_right($right);
        #$action->set_right(new Bi::Expression::Function('std_', [ $right ]));
        $action->validate;
    
        return $action;
    } else {
        return ();
    }
}

=item B<_create_jacobian_actions>(I<node>, I<J>, I<vars>, I<J_vars>)

=cut
sub _create_jacobian_actions {
    my $self= shift;
    my $node = shift;
    my $J = shift;
    my $vars = shift;
    my $J_vars = shift;
    
    my @results;
    
    # Jacobian actions
    my ($dfdxs, $xs) = $node->jacobian;  # nonzero partial derivatives
        
    # the default Jacobian matrix is all zeros except for the block
    # corresponding to noise and state variables, which is an identity matrix,
    # as the absence of an action on a state variable x implies that that
    # state variable does not change, i.e. x <- x.
    my $NR = $self->{_NR};
    my $ND = $self->{_ND};
    my $NO = $self->{_NO};
    my $J0 = new Bi::Expression::Matrix($NR + $ND + $NO, $NR + $ND + $NO);
    $J0->clear;
    $J0->subrange(0, $NR + $ND, 0, $NR + $ND)->ident;

    # ...but for the current action, the default needs to be zero
	my $j = find($vars, $node->get_left->get_var);
	assert ($j >= 0) if DEBUG;
    $J0->set($j, $j, new Bi::Expression::Literal(0.0));
    
    my @Js;
    for (my $k = 0; $k < @$dfdxs; ++$k) {
    	my $dfdx = $dfdxs->[$k];
    	my $x = $xs->[$k];
    	my $i = find($vars, $x->get_var);
		
		if ($i >= 0) {
        	my $Jk = new Bi::Expression::Matrix($J0->size1, $J0->size2);
        	$J0->set($i, $j, new Bi::Expression::Literal(0.0));
        	$Jk->set($i, $j, $dfdx);
        	
        	# local copy of $J with correct indexing
        	my $Jl = $J->clone;
        	for (my $row = 0; $row < $Jl->size1; ++$row) {
        	    my $val = $Jl->get($row, $i);
        	    if ($val->isa('Bi::Expression::VarIdentifier')) {
        	        my @indexes = map { $_->clone } ($val->get_indexes->[0], @{$x->get_indexes});
        	        $val->set_indexes(\@indexes);
        	    }
        	}
        	push(@Js, $Jl*$Jk);
		}
    }
    unshift(@Js, $J*$J0);
        
    # the new Jacobian $J is now the sum of all the matrices in @Js
    my $J_old = $J->clone;
    $J->clear;
    foreach my $Jk (@Js) {
    	$J->assign($J + $Jk);
    }
        
    # construct actions for non-trivial entries in $J
    for (my $j = 0; $j < $J->size2; ++$j) {
        for (my $i = 0; $i < $J->size1; ++$i) {
        	my $old_right = $J_old->get($i, $j);
            my $right = $J->get($i, $j);
            
            if (!$right->equals($old_right)) {
                my $left = $J_vars->get($i, $j)->clone;
                my @indexes = map { $_->clone } ($left->get_indexes->[0], @{$node->get_left->get_indexes});
                my @aliases = (new Bi::Model::DimAlias(undef, $left->get_indexes->[0]), @{$node->get_aliases});

                $left->set_indexes(\@indexes);
                
                my $action = new Bi::Action;
                $action->set_aliases(\@aliases);
                $action->set_left($left);
                if ($node->get_name eq 'ode_') {
                 	$action->set_op('=');
                   	$action->set_right(new Bi::Expression::Function('ode_', [ $right ]));
                } else {
                    $action->set_op('<-');
                    $action->set_right($right);
                }
                $action->validate;
                push(@results, $action);

                $J->set($i, $j, $left->clone);
            }
        }
    }
    return @results[0..$#results];
}

=back

=head1 CLASS METHODS

=over 4

=item B<_add_F_vars>(I<model>)

Add variables that will hold Jacobian terms for the transition model.

=cut
sub _add_F_vars {
    my $model = shift;
    
    my $types1 = ['noise', 'state'];
    my $types2 = ['noise', 'state'];
    
    return _add_vars($model, 'F', $types1, $types2);
}

=item B<_add_G_vars>(I<model>)

Add variables that will hold Jacobian terms for the observation model.

=cut
sub _add_G_vars {
    my $model = shift;
    
    my $types1 = ['noise', 'state'];
    my $types2 = ['obs'];
    
    return _add_vars($model, 'G', $types1, $types2);
}

=item B<_add_Q_vars>(I<model>)

Add variables that will hold state square-root of covariance matrix terms.

=cut
sub _add_Q_vars {
    my $model = shift;
    
    my $types1 = ['noise', 'state'];
    my $types2 = ['noise', 'state'];
    
    return _add_vars($model, 'Q', $types1, $types2);
}

=item B<_add_R_vars>(I<model>)

Add variables that will hold observation square-root of covariance matrix terms.

=cut
sub _add_R_vars {
    my $model = shift;
    
    my $types1 = ['obs'];
    my $types2 = ['obs'];
    
    return _add_vars($model, 'R', $types1, $types2);
}

=item B<_add_vars>(I<model>, I<prefix>, I<rows>, I<vars2>)

=cut
sub _add_vars {
    my $model = shift;
    my $prefix = shift;
    my $types1 = shift;
    my $types2 = shift;
    
    my $vars1 = $model->get_all_vars($types1);
    my $vars2 = $model->get_all_vars($types2);
    my $start = $model->get_size('state_aux_');
    my $rows = $model->get_size($types1);
    my $size = 0;
        
    my $A_vars = new Bi::Expression::Matrix(scalar(@$vars1), scalar(@$vars2));
    
    for (my $j = 0; $j < @$vars2; ++$j) {
   	    my $var2 = $vars2->[$j];   	    
        my $A_var = $model->add_column_var($prefix, $rows, $var2);
        $model->push_var($A_var);
        $size += $A_var->get_size;

        my $offset = 0;
        for (my $i = 0; $i < @$vars1; ++$i) {
            my $len = $vars1->[$i]->get_size;
            my $ranges = $A_var->gen_ranges;
            $ranges->[0] = new Bi::Expression::Range(new Bi::Expression::IntegerLiteral($offset), new Bi::Expression::IntegerLiteral($offset + $len - 1));
            $A_vars->set($i, $j, new Bi::Expression::VarIdentifier($A_var, $ranges));
            $offset += $len;
        }
    }
    $model->set_named_arg("${prefix}_start_", new Bi::Expression::Literal($start));
    $model->set_named_arg("${prefix}_size_", new Bi::Expression::Literal($size));
    
    return $A_vars;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
