=head1 NAME

Bi::Visitor::StaticReplacer - visitor for replacing static subexpressions
by moving into precompute block.

=head1 SYNOPSIS

    use Bi::Visitor::StaticExtractor;
    use Bi::Visitor::StaticReplacer;
    
    my ($lefts, $rights) = Bi::Visitor::StaticExtractor->evaluate($model);
    Bi::Visitor::StaticReplacer->evaluate($model, $lefts, $rights);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::StaticReplacer;

use parent 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<model>, I<actions>)

Evaluate.

=over 4

=item I<model>

L<Bi::Model> object.

=item I<actions>

Array ref of L<Bi::Action> objects returned by
L<Bi::Visitor::StaticEvaluator>.

=back

Replaces any occurrences of the right hand side of each action with the
left hand side (target).

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;
    my $lefts = shift;
    my $rights = shift;

    my $self = new Bi::Visitor; 
    bless $self, $class;

    foreach my $name ('transition', 'lookahead_transition', 'observation', 'lookahead_observation', 'bridge') {
        my $block = $model->get_block($name);
        if (defined $block) {
    	    $block->accept($self, $model, $lefts, $rights);
        }
    }
}

=item B<visit_after>(I<node>, I<model>, I<lefts>, I<rights>, I<actions>)

Visit node.

=cut
sub visit_before {
    my $self = shift;
    my $node = shift;
    my $model = shift;
    my $lefts = shift;
    my $rights = shift;

    if ($node->isa('Bi::Expression')) {
        for (my $i = 0; $i < @$lefts; ++$i) {
            my $left = $lefts->[$i];
            my $right = $rights->[$i];
            if ($node->equals($right)) {
                $node = $left->clone;
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
