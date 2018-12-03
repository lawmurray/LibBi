=head1 NAME

Bi::Visitor::InitialToParamTransformer - visitor for augmenting the
parameters of a model with its initial conditions.

=head1 SYNOPSIS

    use Bi::Visitor::InitialToParamTransformer;
    Bi::Visitor::InitialToParamTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=cut

package Bi::Visitor::InitialToParamTransformer;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

use Bi::Visitor::TargetReplacer;
use Bi::Visitor::VarReplacer;

=head1 METHODS

=over 4

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

    my $parameter_block = $model->get_block('parameter');
    my $initial_block = $model->get_block('initial');
    my $proposal_parameter_block = $model->get_block('proposal_parameter');
    my $proposal_initial_block = $model->get_block('proposal_initial');

    # move the contents of the 'initial' block to the end of the 'parameter'
    # block
    $parameter_block->push_children($initial_block->get_children);
    $initial_block->clear;

    # move the contents of the 'proposal_initial' block to the end of the
    # 'proposal_parameter' block
    $proposal_parameter_block->push_children($proposal_initial_block->get_children);
    $proposal_initial_block->clear;
   
    foreach my $state (@{$model->get_all_vars('state')}) {
        # create a param variable to hold the initial value of this state
        # variable
        my $param = $state->clone;
        $param->set_name($state->get_name . "_0_");
        $param->set_type('param');
        $param->set_named_arg('has_input', new Bi::Expression::IntegerLiteral(1));
        $param->set_named_arg('has_output', new Bi::Expression::IntegerLiteral(0));
        $param->set_named_arg('input_name', new Bi::Expression::StringLiteral($state->get_name));
        $param->set_named_arg('output_name', new Bi::Expression::StringLiteral($param->get_name));
        $model->push_var($param);

        # replace the state variable with the param variable in the
        # 'parameter' and 'proposal_parameter' blocks
        Bi::Visitor::TargetReplacer->evaluate($parameter_block, $state, $param);
        Bi::Visitor::TargetReplacer->evaluate($proposal_parameter_block, $state, $param);
        Bi::Visitor::VarReplacer->evaluate($parameter_block, $state, $param);
        Bi::Visitor::VarReplacer->evaluate($proposal_parameter_block, $state, $param);

        # insert actions into the 'initial' and 'proposal_initial' blocks to
        # copy the param variables into the state variables
        my $action = new Bi::Action;
        my $left = new Bi::Expression::VarIdentifier($state);
        my $right = new Bi::Expression::VarIdentifier($param);
        $action->set_left($left);
        $action->set_op('<-');
        $action->set_right($right);
        $action->validate;
        
        $initial_block->push_child($action->clone);
        $proposal_initial_block->push_child($action->clone);
    }
}

=item B<visit_after>(I<node>)

Visit node of model.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

