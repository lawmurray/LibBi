=head1 NAME

Bi::Visitor::IsConst - is expression a constant expression?

=head1 SYNOPSIS

    use Bi::Visitor::IsConst;
    
    if (Bi::Visitor::IsConst->evaluate($expr)) {
        ...
    }

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::IsConst;

use parent 'Bi::Visitor';
use warnings;
use strict;

use List::Util;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> Expression.

=back

Returns true if I<expr> evaluates to an element, false otherwise.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    my $arg = 1;
    $expr->accept($self, \$arg);
    
    return $arg;
}

=item B<visit_after>(I<node>)

Visit node.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $arg = shift;
    
    if ($node->isa('Bi::Expression::Function') && !$node->is_math) {
        $$arg = 0;
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
        $$arg = 0;
    } elsif ($node->isa('Bi::Expression::DimAliasIdentifier')) {
    	$$arg = 0;
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $$arg = $node->get_inline->get_expr->is_const;
    }

    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

