=head1 NAME

Bi::Visitor::ExtendedTransformer - visitor for constructing linearised version of a
transition block.

=head1 SYNOPSIS

    use Bi::Visitor::ExtendedTransformer;
    Bi::Visitor::ExtendedTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::ExtendedTransformer;

use base 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

use Bi::Jacobian;
use Bi::Utility qw(find);

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

    my $self = {};
    bless $self, $class;

    _add_F_vars($model);
    _add_G_vars($model);
    _add_Q_vars($model);
    _add_R_vars($model);
    
    _transform($model->get_block('initial'), $model, ['noise', 'state']);
    _transform($model->get_block('transition'), $model, ['noise', 'state']);
    _transform($model->get_block('observation'), $model, ['state', 'obs']);
}

sub _transform {
    my $block = shift;
    my $model = shift;
    my $types = shift;
    
    my $vars = $model->get_vars($types);
    my $N = $model->num_vars($types);
    my $J_commit = new Bi::Expression::Matrix($N, $N);
    my $J_new = new Bi::Expression::Matrix($N, $N);
    $J_commit->ident;
    $J_new->ident;
    
    _augment($block, $model, $vars, $J_commit, $J_new);
}

sub _augment {
    my $block = shift;
    my $model = shift;
    my $vars = shift;
    my $J_commit = shift;
    my $J_new = shift;

    foreach my $subblock (@{$block->get_blocks}) {
        _augment($subblock, $model, $vars, $J_commit, $J_new);
    }

    my $N = $J_commit->num_cols;
    my $mu = new Bi::Expression::Vector($N);
    my $S = new Bi::Expression::Matrix($N, $N);
    my $J = new Bi::Expression::Matrix($N, $N);

    # get and then clear actions, will be replaced
    my $actions = $block->get_actions;
    $block->set_actions([]);
    
    foreach my $action (@$actions) {
        # search for index that corresponds to the target of this action
        my $j = find($vars, $action->get_target->get_var);
        assert ($j >= 0);
        
        # mean
        my $mean = $action->mean;
        if (defined $mean) {
            $mu->set($j, $mean);
        }

        # square-root covariance
        my $std = $action->std;
        if (defined $std) {
            $S->set($j, $j, $std);
        }
                
        # Jacobian
        my ($ds, $refs) = $action->jacobian;
        for (my $k = 0; $k < @$ds; ++$k) {
            my $ref = $refs->[$k];
            my $d = $ds->[$k];
            my $i = find($vars, $ref->get_var);
            if ($i >= 0) {
                $J->set($i, $j, $d);
                $J_new->set($i, $j, $d->clone);
            }
        }
    }
    _inline($model, $J);
    $block->add_mean_actions($model, $vars, $mu);
    $block->add_std_actions($model, $vars, $S);
    $block->add_jacobian_actions($model, $vars, $J_commit, $J);

    if ($block->get_commit) {
        $J_commit->swap(Bi::Jacobian::commit($model, $vars, $J_commit*$J_new));
        $J_new->ident;
    }
}

=item B<_add_F_vars>(I<model>)

Add variables that will hold Jacobian terms for the transition model.

=cut
sub _add_F_vars {
    my $model = shift;
    
    my $start = $model->get_size('state_aux_');
    my $size = 0;
    my $vars1 = $model->get_vars(['noise', 'state']);
    my $vars2 = $model->get_vars('state');
    
    foreach my $var2 (@$vars2) {
        foreach my $var1 (@$vars1) {
            my $j_var = $model->add_pair_var('F', $var1, $var2);
            $model->add_var($j_var);
            $size += $j_var->get_size;
        }
    }
    
    $model->set_named_arg('F_start_', new Bi::Expression::Literal($start));
    $model->set_named_arg('F_size_', new Bi::Expression::Literal($size));
}

=item B<_add_G_vars>(I<model>)

Add variables that will hold Jacobian terms for the observation model.

=cut
sub _add_G_vars {
    my $model = shift;
    
    my $start = $model->get_size('state_aux_');
    my $size = 0;
    my $vars1 = $model->get_vars(['noise', 'state']);
    my $vars2 = $model->get_vars('obs');
    
    foreach my $var2 (@$vars2) {
        foreach my $var1 (@$vars1) {
            my $j_var = $model->add_pair_var('G', $var1, $var2);
            $model->add_var($j_var);
            $size += $j_var->get_size;
        }
    }
    
    $model->set_named_arg('G_start_', new Bi::Expression::Literal($start));
    $model->set_named_arg('G_size_', new Bi::Expression::Literal($size));
}

=item B<_add_Q_vars>(I<model>)

Add variables that will hold state square-root of covariance matrix terms.

=cut
sub _add_Q_vars {
    my $model = shift;
    
    my $start = $model->get_size('state_aux_');
    my $size = 0;
    my $vars = $model->get_vars(['noise', 'state']);
    
    foreach my $var2 (@$vars) {
        foreach my $var1 (@$vars) {
            my $j_var = $model->add_pair_var('Q', $var1, $var2);
            $model->add_var($j_var);
            $size += $j_var->get_size;
        }
    }
    
    $model->set_named_arg('Q_start_', new Bi::Expression::Literal($start));
    $model->set_named_arg('Q_size_', new Bi::Expression::Literal($size));
    
}

=item B<_add_R_vars>(I<model>)

Add variables that will hold observation square-root of covariance matrix terms.

=cut
sub _add_R_vars {
    my $model = shift;
    
    my $start = $model->get_size('state_aux_');
    my $size = 0;
    my $vars = $model->get_vars('obs');
    
    foreach my $var2 (@$vars) {
        foreach my $var1 (@$vars) {
            my $j_var = $model->add_pair_var('R', $var1, $var2);
            $model->add_var($j_var);
            $size += $j_var->get_size;
        }
    }
    
    $model->set_named_arg('R_start_', new Bi::Expression::Literal($start));
    $model->set_named_arg('R_size_', new Bi::Expression::Literal($size));
    
}

=item B<_inline>(I<model>, I<J>)

Add inlines for Jacobian terms.

=cut
sub _inline {
    my $model = shift;
    my $J = shift;
    
    for (my $j = 0; $j < $J->num_cols; ++$j) {
        for (my $i = 0; $i < $J->num_rows; ++$i) {
            my $expr = $J->get($i, $j);
            if (defined $expr && !$expr->is_one) {
                my $inline = $model->lookup_inline($expr);
                if (!defined $inline) {
                    $inline = new Bi::Model::Inline($model->tmp_inline, $expr);
                    $model->add_inline($inline);
                }
                
                my $ref = new Bi::Expression::InlineIdentifier($inline);
                $J->set($i, $j, $ref);
            }
        }
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
