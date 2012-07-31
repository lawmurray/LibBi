=head1 NAME

Bi::Visitor::StaticEvaluate - visitor for adding precomputation of static
expressions to parameter block.

=head1 SYNOPSIS

    use Bi::Visitor::StaticExtractor;
    use Bi::Visitor::StaticEvaluator;
    use Bi::Visitor::StaticReplacer;
    
    my $extracts = Bi::Visitor::StaticExtractor->evaluate($model);
    my $actions = Bi::Visitor::StaticEvaluator->evaluate($model, $extracts);
    Bi::Visitor::StaticReplacer->evaluate($model, $actions);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::StaticEvaluator;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<model>, I<extracts>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=item I<item>

=item I<extracts> List of L<Bi::Expression> giving static subexpressions
extracted by L<Bi::Visitor::StaticExtractor>.

=back

Constructs actions to perform precomputation of extracted static
subexpressions, inserts these into model, and returns an array ref of them.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;
    my $extracts = shift;

    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    # create actions from extracted subexpressions
    my $actions = [];
    my $var;
    my $target;
    my $action;
    foreach my $extract (@$extracts) {
        $var = new Bi::Model::ParamAux($model->tmp_var, $extract->get_dims);
        $model->add_var($var);
        $target = new Bi::Expression::VarIdentifier($var);
        
        $action = new Bi::Model::Action($model->next_action_id, $target, '<-', $extract->clone);
        push(@$actions, $action);
    }

    # create precompute block
    if (@$actions) {
        my $block = new Bi::Model::Block($model->next_block_id);
        $block->push_actions($actions);
        $block->set_commit(1);
        
        $model->accept($self, $model, $block);
    }
    return $actions;
}

=item B<visit>(I<node>, I<actions>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $model = shift;
    my $block = shift;

    if ($node->isa('Bi::Model::Block')) {
        if ($node->get_name eq 'parameter' || $node->get_name eq 'proposal_parameter') {
            $node->sink_actions($model);
            if ($node->num_blocks > 0) {
               $node->get_block($node->num_blocks - 1)->set_commit(1);
            }
            $node->push_block($block->clone($model));
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
