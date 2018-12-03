=head1 NAME

Bi::Visitor::IsStatic - is expression a static expression?

=head1 SYNOPSIS

    use Bi::Visitor::IsStatic;
    
    if (Bi::Visitor::IsStatic->evaluate($expr)) {
        ...
    }

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::IsStatic;

use parent 'Bi::Visitor';
use warnings;
use strict;

use List::Util;
use Bi::Utility qw(contains);

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
    } elsif ($node->isa('Bi::Expression::VarIdentifier') &&
            !contains(['param', 'param_aux_'], $node->get_var->get_type)) {
    	$$arg = 0;
    } elsif ($node->isa('Bi::Expression::DimAliasIdentifier')) {
    	$$arg = 0;
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $$arg = $node->get_inline->get_expr->is_static;
    }

    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

