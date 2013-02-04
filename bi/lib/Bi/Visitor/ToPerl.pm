=head1 NAME

Bi::Visitor::ToPerl - visitor for translating expression into a Perl
string.

=head1 SYNOPSIS

    use Bi::Visitor::ToPerl;
    $str = Bi::Visitor::ToPerl->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::ToPerl;

use base 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;

use Bi::Expression;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns the expression as a Perl string.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    assert (defined $expr) if DEBUG;
    
    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    my $args = [];    
    $expr->accept($self, $args);
    return pop(@$args);
}

=item B<visit>(I<node>)

Visit node of expression tree.

=cut
sub visit {
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
        my $name = $node->get_name;
        if ($name eq 'pow') {
            $str = $args[0] . '**' . $args[1];
        } else {
            $str = $node->get_name . '(' . join(', ', @args) . ')';
        }
    } elsif ($node->isa('Bi::Expression::ConstIdentifier')) {
        $node->get_const->get_expr->accept($self, $args);
        my $expr = pop(@$args);
        $str = '(' . $expr . ')';
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->get_expr->accept($self, $args);
        my $expr = pop(@$args);
        $str = '(' . $expr . ')';
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
        $str = '$' . $node->get_var->get_name;
        if ($node->num_indexes) {
            my @indexes = splice(@$args, -$node->num_indexes, $node->num_indexes);
            $str .= '_' . join('_', @indexes);
        }
    } elsif ($node->isa('Bi::Expression::DimAliasIdentifier')) {
        $str = $node->get_alias->get_name;
    } elsif ($node->isa('Bi::Expression::Literal') || $node->isa('Bi::Expression::IntegerLiteral')) {
        $str = $node->get_value;
    } elsif ($node->isa('Bi::Expression::StringLiteral')) {
        $str = $node->get_value;
    } elsif ($node->isa('Bi::Expression::Index')) {
    	my @exprs = pop(@$args);
        $str = $exprs[0];
    } elsif ($node->isa('Bi::Expression::Range')) {
        my @exprs = splice(@$args, -2);
        $str = "$exprs[0]..$exprs[1]";
    } elsif ($node->isa('Bi::Expression::TernaryOperator')) {
        my @exprs = splice(@$args, -3);
        $str = '(' . $exprs[0] . ' ?' . $exprs[1] . ' : ' . $exprs[2] . ')';
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

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
