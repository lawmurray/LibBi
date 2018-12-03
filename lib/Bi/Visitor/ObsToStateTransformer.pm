=head1 NAME

Bi::Visitor::ObsToStateTransformer - visitor for augmenting the state of a
model with observations, useful for generating synthetic data sets.

=head1 SYNOPSIS

    use Bi::Visitor::ObsToStateTransformer;
    Bi::Visitor::ObsToStateTransformer->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=cut

package Bi::Visitor::ObsToStateTransformer;

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

    my $observation_block = $model->get_block('observation');
    my $transition_block = $model->get_block('transition');
    my $initial_block = $model->get_block('initial');

    my $lookahead_observation_block = $model->get_block('lookahead_observation');
    my $lookahead_transition_block = $model->get_block('lookahead_transition');

    # moves the contents of the 'observation' block to the end of both the
    # 'transition' and 'initial' blocks
    $transition_block->push_children($observation_block->clone->get_children);
    $initial_block->push_children($observation_block->clone->get_children);
    $observation_block->clear;

    # move the contents of the 'lookahead_observation' block to the end of
    # the 'lookahead_transition' block
    $lookahead_transition_block->push_children($lookahead_observation_block->get_children);
    $lookahead_observation_block->clear;
        
    # change obs variables to state variables    
    $model->accept($self);
}

=item B<visit_after>(I<node>)

Visit node of model.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;

    if ($node->isa('Bi::Model::Var') && $node->get_type eq 'obs') {
        $node->set_type('state');
    }
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

