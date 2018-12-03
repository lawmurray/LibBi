=head1 NAME

Bi::Visitor::Wrapper - visitor for wrapping actions in their preferred parent
block type.

=head1 SYNOPSIS

    use Bi::Visitor::Wrapper;
    Bi::Visitor::Wrapper->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::Wrapper;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Graph;
use Carp::Assert;
use Bi::Utility qw(set_intersect push_unique);

=item B<evaluate>(I<model>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=back

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = new Bi::Visitor;
    bless $self, $class;
    
    foreach my $topblock (@{$model->get_children}) {
        _wrap($topblock);
    }
}

=item B<_wrap>

=cut
sub _wrap {
    my $node = shift;

    # recurse
    foreach my $child (@{$node->get_blocks}) {
        _wrap($child);
    }

    # build directed graph to represent dependencies between children
    my $graph = new Graph(directed => 1, refvertexed => 1);
    foreach my $child (@{$node->get_children}) {
        my @vertices = $graph->vertices;
        $graph->add_vertex($child);

        my $vars = [ @{$child->get_all_left_vars}, @{$child->get_all_right_vars} ];
        foreach my $vertex (@vertices) {
            my $left_vars = $vertex->get_all_left_vars;
            if (@{set_intersect($vars, $left_vars)} > 0) {
                $graph->add_edge($vertex, $child);
            }
        }
    }
    $node->set_children([]);    
    
    # wrap actions in blocks, combining where there are no dependencies
    while ($graph->vertices > 0) {
        # children for which all dependencies are satisfied, heuristic is to
        # handle them in the order given in the model specification
        my @vertices = sort { $a->get_id <=> $b->get_id }
                $graph->predecessorless_vertices;

        # handle the first child with satisfied dependencies
        my $block;
        my $vertex = $vertices[0];
        if ($vertex->isa('Bi::Block')) {
            # is a block already
            $block = $vertex;
            $node->push_child($block);
        } else {
            # is an action
            assert($vertex->isa('Bi::Action')) if DEBUG;
            
            if (defined $node->get_name && $node->get_name eq $vertex->get_parent) {
                # already in block of preferred parent type
                $block = $node;
                $node->push_child($vertex);
            } else {
                # wrap in block of preferred parent type
                $block = new Bi::Block;
                $block->set_name($vertex->get_parent);
                $block->push_child($vertex);
                $node->push_child($block);
            }
        }
        $graph->delete_vertex($vertex);
        
        # greedily add any other children with satisfied dependencies to this
        # block, if it is of their preferred parent type
        foreach $vertex (@vertices[1..$#vertices]) {
            if ($vertex->isa('Bi::Action') && $vertex->can_combine &&
                    $vertex->get_parent eq $block->get_name) {
                $block->push_child($vertex);
                $graph->delete_vertex($vertex);
            }
        }
        $block->validate;
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

