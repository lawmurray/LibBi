=head1 NAME

Bi::Visitor::StaticReplacer - visitor for replacing static subexpressions
by moving into precompute block.

=head1 SYNOPSIS

    use Bi::Visitor::StaticExtractor;
    use Bi::Visitor::StaticReplacer;
    
    $extracts = Bi::Visitor::StaticExtractor-evaluate($model);
    Bi::Visitor::StaticReplacer->evaluate($model, $extracts);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::StaticReplacer;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<model>, I<extracts>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=item I<item>

=item I<extracts> List of L<Bi::Expression> subexpressions to replace.

=back

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;
    my $item = shift;
    my $extracts = shift;

    # create actions from extracted subexpressions
    my @actions;
    my $var;
    my $target;
    my $action;
    foreach my $extract (@$extracts) {
        $var = new Bi::Model::ParamAux($model->tmp_var, $extract->get_dims);
        $model->add_var($var);
        $target = new Bi::Expression::VarIdentifier($var);
        
        $action = new Bi::Model::Action($model->next_action_id, $target, '<-', $extract->clone);
        push(@actions, $action);
    }

    # create precompute block
    my $block_name = 'parameter_post_';
    my $block;
    if ($model->is_block($block_name)) {
        $block = new Bi::Model::Block($model->next_block_id);
        $model->get_block($block_name)->push_block($block);
    } else {
        $block = new Bi::Model::Block($model->next_block_id, $block_name);
        $model->add_block($block);
    }
    $block->push_actions(\@actions);

    my $self = new Bi::Visitor; 
    bless $self, $class;

    $item->accept($self, \@actions);
}

=item B<visit>(I<node>, I<actions>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $actions = shift;

    if ($node->isa('Bi::Expression')) {
        foreach my $action (@$actions) {
            if ($node->equals($action->get_named_arg('expr'))) {
                $node = $action->get_target->clone;
                last;
            }
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
