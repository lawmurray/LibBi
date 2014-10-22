=head1 NAME

Bi::Visitor::StaticExtractor - visitor for extracting static subexpressions.

=head1 SYNOPSIS

    use Bi::Visitor::StaticExtractor;
    
    my ($lefts, $rights) = Bi::Visitor::StaticExtractor->evaluate($model);
    Bi::Visitor::StaticReplacer->evaluate($model, $lefts, $rights);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::StaticExtractor;

use parent 'Bi::Visitor';
use warnings;
use strict;

use List::Util qw(reduce);
use Carp::Assert;

use Bi::Utility qw(unique);

=item B<evaluate>(I<model>)

Constructs actions to perform precomputation of extracted static
subexpressions, inserts these into model, and returns two array refs: the
second gives the extracted expressions, and the first the variable references
to replace them with.

=cut
sub evaluate {
    my $class = shift;
    my $model = shift;

    my $self = new Bi::Visitor; 
    bless $self, $class;

    # extract expressions
    my $statics = [];
    my $extracts = [];    
    foreach my $name ('transition', 'lookahead_transition', 'observation', 'lookahead_observation') {
	    $model->get_block($name)->accept($self, $statics, $extracts);
    }
    $extracts = unique($extracts);
    @$extracts = map { ($_->is_const || $_->isa('Bi::Expression::VarIdentifier')) ? () : $_ } @$extracts;

    # insert actions
    my $lefts = [];
    my $rights = [];
    
    foreach my $extract (@$extracts) {
    	my $dims = [ map { $model->lookup_dim($_) } @{$extract->get_shape->get_sizes} ];
    	my $var = new Bi::Model::Var('param_aux_', undef, $dims, [], {
            'has_input' => new Bi::Expression::IntegerLiteral(0),
            'has_output' => new Bi::Expression::IntegerLiteral(0)
        });
        $model->push_var($var);
        
        my $left = new Bi::Expression::VarIdentifier($var, $var->gen_ranges);
        my $right = $extract;
        
        push(@$lefts, $left);
        push(@$rights, $right);

        my $action = new Bi::Action;
        $action->set_aliases($var->gen_aliases);
        $action->set_left($left);
        $action->set_op('<-');
        $action->set_right($right);
        $action->validate;

        $model->get_block('parameter')->push_child($action);
    }
    
    return ($lefts, $rights);
}

=item B<visit_after>(I<node>)

Visit node.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $statics = shift;
    my $extracts = shift;
    
    my $is_static = 0;
    my $num_statics = 0;
    
    if ($node->isa('Bi::Action')) {
        my $num_args = $node->num_args + $node->num_named_args;
        my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = 0;
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::BinaryOperator')) {
        my @statics = splice(@$statics, -2);
        $is_static = reduce { $a && $b } (1, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::Function')) {
        my $num_args = $node->num_args + $node->num_named_args;
        my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = reduce { $a && $b } (1, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $is_static = $node->get_inline->get_expr->is_static;
    } elsif ($node->isa('Bi::Expression::Index')) {
    	my $num_args = 1;
    	my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = reduce { $a && $b } (1, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::Index')) {
    	my $num_args = 1;
    	my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = reduce { $a && $b } (1, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::Range')) {
    	my $num_args = 2;
    	my @statics = splice(@$statics, -$num_args, $num_args);
        $is_static = reduce { $a && $b } (1, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
        my @statics = splice(@$statics, -@{$node->get_indexes}, scalar(@{$node->get_indexes}));
        $is_static = reduce { $a && $b } ($node->is_static, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::TernaryOperator')) {
        my @statics = splice(@$statics, -3);
        $is_static = reduce { $a && $b } (1, @statics);
        $num_statics = reduce { $a + $b } (0, @statics);
    } elsif ($node->isa('Bi::Expression::UnaryOperator')) {
        $is_static = pop(@$statics);
        $num_statics = $is_static;
    } elsif ($node->isa('Bi::Expression::ConstIdentifier')) {
        $is_static = 1;
    } elsif ($node->isa('Bi::Expression::Literal')) {
        $is_static = 1;
    } elsif ($node->isa('Bi::Expression::IntegerLiteral')) {
        $is_static = 1;
    } elsif ($node->isa('Bi::Expression::StringLiteral')) {
        $is_static = 1;
    }
    
    push(@$statics, $is_static);
    if ($is_static) {
        if ($num_statics > 0) {
            splice(@$extracts, -$num_statics, $num_statics, $node->clone);
        } else {
        	assert($node->is_static) if DEBUG;
            push(@$extracts, $node->clone);
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
