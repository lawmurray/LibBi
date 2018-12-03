=head1 NAME

Bi::Visitor::IteratedSmoothingTransformer - visitor for augmenting the state of a
model with its parameters.

=head1 SYNOPSIS

    use Bi::Visitor::IteratedSmoothingTransformer;
    Bi::Visitor::IteratedSmoothingTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=cut

package Bi::Visitor::IteratedSmoothingTransformer;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

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
    my $transition_block = $model->get_block('transition');
    my $lookahead_observation_block = $model->get_block('lookahead_observation');
    my $lookahead_transition_block = $model->get_block('lookahead_transition');
    # keep the names of the original parameters
    my @originalparameters = @{$model->get_all_vars('param')};
    # clear original parameter block
    $parameter_block->clear;
    # clear proposals
    $proposal_initial_block->clear;    
    $proposal_parameter_block->clear;
    # clear lookaheads and proposal_initial
    $lookahead_observation_block->clear;
    $lookahead_transition_block->clear;
    # self-acceptance is defined as affirmation or acceptance of self in spite of weaknesses or deficiencies.
    $model->accept($self);
    # add iterated smoothing algorithmic parameters model parameters
    my $tau = new Bi::Model::Var('param', 'tau_');
    $model->push_var($tau);
    # set a default value for tau (equivalently a dirac mass prior distribution)
    my $actionSetTau = new Bi::Action;
    my $left = new Bi::Expression::VarIdentifier($tau);
    my $right = new Bi::Expression::Literal(1.0);
    $actionSetTau->set_left($left);
    $actionSetTau->set_op('<-');
    $actionSetTau->set_right($right);
    $actionSetTau->validate;
    $parameter_block->unshift_child($actionSetTau->clone);
    # for each of the original parameter...
    foreach my $param (@originalparameters) {
        # define a new parameter to be used as the mean of the original parameter's distribution
        my $param0 = new Bi::Model::Var('param', $param->get_name . "_0_");
        $model->push_var($param0);
        # set a default value to it:
        # $param0 <- 1.0
        my $actionSetParam0 = new Bi::Action;
        my $left = new Bi::Expression::VarIdentifier($param0);
        my $right = new Bi::Expression::Literal(1.0);
        $actionSetParam0->set_left($left);
        $actionSetParam0->set_op('<-');
        $actionSetParam0->set_right($right);
        $actionSetParam0->validate;
        $parameter_block->unshift_child($actionSetParam0->clone);
        # create an action like:
        # $param ~ gaussian(mean = $param_0_, sd = $tau)
        my $actionGaussian = new Bi::Action;
        $left = new Bi::Expression::VarIdentifier($param);
        my $name = 'gaussian';
        my $named_args = {
            'mean' =>new Bi::Expression::VarIdentifier($param0),
            'std' => new Bi::Expression::VarIdentifier($tau)
        };
        $actionGaussian->set_left($left);
        $actionGaussian->set_op('~');
        $actionGaussian->set_name($name);
        $actionGaussian->set_named_args($named_args);
        $actionGaussian->validate;
        # put this in the initial and transition block, ie
        # $param ~iid gaussian($param0, $tau) at every time step
        $initial_block->unshift_child($actionGaussian->clone);
        $transition_block->unshift_child($actionGaussian->clone);
    }
    
}

=item B<visit_after>(I<node>)

Visit node of model.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;

    if ($node->isa('Bi::Model::Var') && $node->get_type eq 'param') {
        $node->set_type('state');
    }
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au> & Pierre Jacob

