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
use FindBin qw($Bin);
use File::Spec;

use Bi::Visitor::ToCpp;

=item B<evaluate>(I<model>)

Construct and evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=back

No return value.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = {};
    bless $self, $class;

    $self->_add_sqrt_vars($model);
    foreach my $name ('initial', 'transition', 'observation') {
        $model->get_block($name)->accept($self, $model, [], []);
    }
}

=item B<visit>(I<node>)

Visit node of model.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $model = shift;
    my $new_blocks = shift;
    my $new_actions = shift;

    if ($node->isa('Bi::Model::Block')) {
        $node->push_blocks($new_blocks);
        $node->push_actions($new_actions);
        @$new_blocks = ();
        @$new_actions = ();
    } elsif ($node->isa('Bi::Model::Action')) {
        # update to square-root covariance
        my $std = $node->std;
        if (defined($std)) {
            my $var_name = $node->get_target->get_var->get_name;
            my $sqrt_var_name = sprintf("S_%s_%s_", $var_name, $var_name);
            my $sqrt_var = $model->get_var($sqrt_var_name);
            my @sqrt_offsets = map { $_->clone } @{$node->get_target->get_offsets};
            my @extra_offsets = map { $_->clone } @{$node->get_target->get_offsets};
            push(@sqrt_offsets, @extra_offsets);
            my $sqrt_target = new Bi::Expression::VarIdentifier($sqrt_var, \@sqrt_offsets);
            
            push(@$new_actions, ref($node)->new_std_action($model->next_action_id, $sqrt_target, $std));
        }
        
        # Jacobian updates to square-root cross-covariances
        my ($Js, $refs) = $node->jacobian;
        if (@$Js) {
            my $block = new Bi::Model::Block($model->next_block_id);
            my @vars;
            foreach my $type ('noise', 'state') {
                push(@vars, @{$model->get_vars($type)});
            }
            
            foreach my $var (@vars) {
                my $sqrt_var_name = sprintf("S_%s_%s_", $node->get_target->get_var->get_name, $var->get_name);
                my $sqrt_var = $model->get_var($sqrt_var_name);
                my @sqrt_offsets = map { $_->clone } @{$node->get_target->get_offsets};

                my @extra_offsets = ();
                if ($var->num_dims > 0) {
                    @extra_offsets = map { new Bi::Expression::Offset("i${_}_", 0) } (1..$var->num_dims);
                }
                
                push(@sqrt_offsets, @extra_offsets);
                my $sqrt_target = new Bi::Expression::VarIdentifier($sqrt_var, \@sqrt_offsets);
                
                my @sqrt_refs;
                foreach my $ref (@$refs) {
                    my @offsets = map { $_->clone } @{$ref->get_offsets};
                    push(@sqrt_refs, new Bi::Expression::VarIdentifier($model->get_var(sprintf("S_%s_%s_", $ref->get_var->get_name, $var->get_name)), \@offsets));
                }
                
                if (@extra_offsets) {
                    map { push(@{$_->get_offsets}, @extra_offsets) } @sqrt_refs;
                }
                
                push(@$new_actions, ref($node)->new_jacobian_action($model->next_action_id, $sqrt_target, $Js, \@sqrt_refs));
            }
        }
        $node = ref($node)->new_mean_action($model->next_action_id, $node->get_target, $node->mean);
    }
    return $node;
}

=item B<_add_sqrt_vars>

Add variables to hold rows of square-root of covariance matrix.

=cut
sub _add_sqrt_vars {
    my $self = shift;
    my $model = shift;
    
    my ($var1, $var2);
    my @vars;
    my $sqrt_var;
    my $start;
    my $size;
    
    # dense
    $start = $model->get_size('state_aux_');
    $size = 0;
    @vars = (@{$model->get_vars('noise')}, @{$model->get_vars('state')}, @{$model->get_vars('obs')});
    foreach $var1 (@vars) {
        foreach $var2 (@vars) {
            $sqrt_var = $self->_create_sqrt_var($model, $var1, $var2);
            $size += $sqrt_var->get_size;
            $model->add_var($sqrt_var);
        }
    }
    $model->set_named_arg('std_start_', new Bi::Expression::Literal($start));
    $model->set_named_arg('std_size_', new Bi::Expression::Literal($size));
}

=item B<_create_sqrt_var>(I<var1>, I<var2>)

Create the variable that will hold the block corresponding to the dependence
of I<var1> on I<var2> in the square-root of covariance matrix.

=cut
sub _create_sqrt_var {
    my $self = shift;
    my $model = shift;
    my $var1 = shift;
    my $var2 = shift;
    
    my $name = 'S_' . $var1->get_name . '_' . $var2->get_name . '_';
    my $dims = [ @{$var1->get_dims}, @{$var2->get_dims} ];
    my $named_args = {
        'io' => new Bi::Expression::Literal(0),
        'tmp' => $var1->get_named_arg('tmp')->clone
    };
    my $sqrt_var = new Bi::Model::StateAux($name, $dims, [], $named_args);
    
    return $sqrt_var;
}

=item B<_add_zero_block>(I<block>, I<types1>, I<types2>)

Add block to start of I<block> that will zero all square-root covariance
variables which involve variables of types I<types1> dependent on variables
of types I<types2>.

=cut
sub _add_zero_block {
    my $self = shift;
    my $model = shift;
    my $block = shift;
    my $types1 = shift;
    my $types2 = shift;
    
    my $zero_block = new Bi::Model::Block($model->next_block_id);
    my $zero_action;
    my $var1;
    my $var2;
    
    my @vars1 = map { $_->get_name !~ /^S_\w+_/ ? $_ : () } @{$model->get_vars($types1)};
    my @vars2 = map { $_->get_name !~ /^S_\w+_/ ? $_ : () } @{$model->get_vars($types2)};
    foreach $var1 (@vars1) {
        foreach $var2 (@vars2) {
            $zero_action = $self->_create_zero_action($model, $var1, $var2);
            $zero_block->push_action($zero_action);
        }
    }
    
    # wrap any loose actions in another block
    if ($block->num_actions > 0) {
        my $child = new Bi::Model::Block($model->next_block_id, 'eval', [], {}, $block->get_actions, []);
        $block->set_actions([]);
        $block->unshift_block($child);
    }
    
    # add zero block to start
    $zero_block->set_commit(1);
    $block->unshift_block($zero_block);
}

=item B<_create_zero_action>(I<var1>, I<var2>)

Create an action that zeros the square-root covariance variable between
I<var1> and I<var2>.

=cut
sub _create_zero_action {
    my $self = shift;
    my $model = shift;
    my $var1 = shift;
    my $var2 = shift;
    
    my $name = 'S_' . $var1->get_name . '_' . $var2->get_name . '_';
    my $var = $model->get_var($name);
    my $target = new Bi::Expression::VarIdentifier($var);
    my $action = new Bi::Model::Action($model->next_action_id, $target, '<-', new Bi::Expression::Literal(0));

    return $action;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
