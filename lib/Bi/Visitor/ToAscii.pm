=head1 NAME

Bi::Visitor::ToAscii - visitor for translating expression into an ASCII
string.

=head1 SYNOPSIS

    use Bi::Visitor::ToAscii;
    $ascii = Bi::Visitor::ToAscii->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::ToAscii;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

use Bi::Expression;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns the expression as an ASCII string.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    my $args = [];    
    $expr->accept($self, $args);
    return pop(@$args);
}

=item B<visit_after>(I<node>)

Visit node of expression tree.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $args = shift;
    my $str;
    
    if ($node->isa('Bi::Expression::BinaryOperator')) {
        my @exprs = splice(@$args, -2);
        my $op = $node->get_op;
        my $space = ($op eq '*' || $op eq '/') ? '' : ' ';
        $str = "($exprs[0]$space$op$space$exprs[1])";
    } elsif ($node->isa('Bi::Expression::Function')) {
        my $num_args = $node->num_args + $node->num_named_args;
        my @args = splice(@$args, -$num_args, $num_args);
        $str = $node->get_name . '(' . join(', ', @args) . ')';
    } elsif ($node->isa('Bi::Expression::ConstIdentifier')) {
        $str = $node->get_const->get_name;
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $str = $node->get_inline->get_name;
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
        $str = $node->get_var->get_name;
        if (@{$node->get_indexes}) {
            my @indexes = splice(@$args, -@{$node->get_indexes}, scalar(@{$node->get_indexes}));
            $str .= '[' . join(',', @indexes) . ']';
        }
    } elsif ($node->isa('Bi::Expression::DimAliasIdentifier')) {
        $str = $node->get_alias->get_name;
    } elsif ($node->isa('Bi::Expression::Literal') || $node->isa('Bi::Expression::IntegerLiteral')) {
        $str = $node->get_value;
    } elsif ($node->isa('Bi::Expression::StringLiteral')) {
        $str = '"' . $node->get_value . '"';
    } elsif ($node->isa('Bi::Expression::Index')) {
        my @exprs = pop(@$args);
        $str = $exprs[0];
    } elsif ($node->isa('Bi::Expression::Range')) {
        my $end = pop(@$args);
        my $start = pop(@$args);
      	$str = "$start:$end";
    } elsif ($node->isa('Bi::Expression::TernaryOperator')) {
        my @exprs = splice(@$args, -3);
        $str = '(' . $exprs[0] . ' ? ' . $exprs[1] . ' : ' . $exprs[2] . ')';
    } elsif ($node->isa('Bi::Expression::UnaryOperator')) {
        my @exprs = pop(@$args);
        $str = $node->get_op . $exprs[0];
    } else {
        die("unsupported expression type\n");
    }
    
    push(@$args, $str);
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
