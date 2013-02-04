=head1 NAME

Bi::Visitor::StaticReplacer - visitor for replacing static subexpressions
by moving into precompute block.

=head1 SYNOPSIS

    use Bi::Visitor::StaticExtractor;
    use Bi::Visitor::StaticEvaluator;
    use Bi::Visitor::StaticReplacer;
    
    my $extracts = Bi::Visitor::StaticExtractor->evaluate($model);
    my $actions = Bi::Visitor::StaticEvaluator->evaluate($model, $actions);
    Bi::Visitor::StaticReplacer->evaluate($model, $actions);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::StaticReplacer;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<model>, I<actions>)

Evaluate.

=over 4

=item I<model> L<Bi::Model> object.

=item I<item>

=item I<actions> Array ref of L<Bi::Model::Action> objects returned by
L<Bi::Visitor::StaticEvaluator>.

=back

Replaces any occurrences of the right hand side of each action with the
left hand side (target).

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;
    my $actions = shift;

    my $self = new Bi::Visitor; 
    bless $self, $class;

    foreach my $name ('transition', 'lookahead_transition', 'observation', 'lookahead_observation') {
        if ($model->is_block($name)) {
    	    $model->get_block($name)->accept($self, $model, $actions);
        }
    }
}

=item B<visit>(I<node>, I<actions>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $model = shift;
    my $actions = shift;

    if ($node->isa('Bi::Expression')) {
        foreach my $action (@$actions) {
            my $expr = $action->get_named_arg('expr');            
            if ($node->equals($expr)) {
                $node = $action->get_left->clone;
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
