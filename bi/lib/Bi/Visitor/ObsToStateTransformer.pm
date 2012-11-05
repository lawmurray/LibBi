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

use base 'Bi::Visitor';
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

    my $observation_block = $model->get_block('observation')->clone($model);
    $observation_block->set_name('eval_');
    $observation_block->set_commit(1);
    
    my $lookahead_observation_block = $model->get_block('lookahead_observation')->clone($model);
    $lookahead_observation_block->set_name('eval_');
    $lookahead_observation_block->set_commit(1);
    
    $model->accept($self, $model, $observation_block, $lookahead_observation_block);
}

=item B<visit>(I<node>)

Visit node of model.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $model = shift;    
    my $observation_block = shift;
    my $lookahead_observation_block = shift;

    if ($node->isa('Bi::Model::Obs')) {
        # replace with state variable
        $node = new Bi::Model::State($node->get_name, $node->get_dims,
            $node->get_args, $node->get_named_args);
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
    	if ($node->get_var->get_type eq 'obs') {
    		my $name = $node->get_var->get_name;
    		$node->set_var($model->get_var($name));
    	} 
    } elsif ($node->isa('Bi::Model::Block')) {
        if ($node->get_name eq 'observation' || $node->get_name eq 'lookahead_observation') {
            $node->clear;
        } elsif ($node->get_name eq 'transition' || $node->get_name eq 'initial' || $node->get_name eq 'proposal_initial') {
            $node->sink_actions($model);
            if ($node->num_blocks > 0) {
                $node->get_block($node->num_blocks - 1)->set_commit(1);
            }
            $node->push_block($observation_block->clone($model));
        } elsif ($node->get_name eq 'lookahead_transition') {
            $node->sink_actions($model);
            if ($node->num_blocks > 0) {
                $node->get_block($node->num_blocks - 1)->set_commit(1);
            }
            $node->push_block($lookahead_observation_block);
        }
    }
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
