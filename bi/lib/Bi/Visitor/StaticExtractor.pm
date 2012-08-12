=head1 NAME

Bi::Visitor::StaticExtractor - visitor for extracting static subexpressions.

=head1 SYNOPSIS

    use Bi::Visitor::StaticExtractor;
    
    my $extracts = Bi::Visitor::StaticExtractor->evaluate($model);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::StaticExtractor;

use base 'Bi::Visitor';
use warnings;
use strict;
use List::Util qw(reduce);

use Bi::Utility qw(unique);

=item B<evaluate>(I<model>)

Evaluate.

Returns array ref of static subexpressions that can be extracted.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = new Bi::Visitor; 
    bless $self, $class;

    my $statics = [];
    my $extracts = [];    
    foreach my $name ('transition', 'lookahead_transition', 'observation', 'lookahead_observation') {
	    $model->get_block($name)->accept($self, $statics, $extracts);
    }
    $extracts = unique($extracts);
    @$extracts = map { ($_->is_const || $_->isa('Bi::Expression::VarIdentifier')) ? () : $_ } @$extracts;

    return $extracts;
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $statics = shift;
    my $extracts = shift;
    
    my $is_static = 1;
    my $num_statics = 0;
    
    if ($node->isa('Bi::Model::Action')) {
        my $num_args = $node->num_args + $node->num_named_args;
        my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = 0;
        $num_statics = reduce { $a + $b } 0, @statics;
    } elsif ($node->isa('Bi::Expression::BinaryOperator')) {
        my @statics = splice(@$statics, -2);
        $is_static = reduce { $a && $b } @statics;
        $num_statics = reduce { $a + $b } @statics;
    } elsif ($node->isa('Bi::Expression::Function')) {
        my $num_args = $node->num_args + $node->num_named_args;
        my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = reduce { $a && $b } 1, @statics;
        $num_statics = reduce { $a + $b } 0, @statics;
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $is_static = $node->get_inline->get_expr->is_static;
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
        my @statics = splice(@$statics, -$node->num_offsets, $node->num_offsets);
        $is_static = reduce { $a && $b } $node->is_static, @statics;
        $num_statics = reduce { $a + $b } 0, @statics;
    } elsif ($node->isa('Bi::Expression::TernaryOperator')) {
        my @statics = splice(@$statics, -3);
        $is_static = reduce { $a && $b } @statics;
        $num_statics = reduce { $a + $b } @statics;
    } elsif ($node->isa('Bi::Expression::UnaryOperator')) {
        $is_static = pop(@$statics);
        $num_statics = $is_static;
    } elsif (!$node->isa('Bi::Expression')) {
        $is_static = 0;
    }
    
    push(@$statics, $is_static);
    if ($is_static) {
        if ($num_statics > 0) {
            splice(@$extracts, -$num_statics, $num_statics, $node);
        } else {
            push(@$extracts, $node);
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
