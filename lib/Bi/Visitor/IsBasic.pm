=head1 NAME

Bi::Visitor::IsBasic - is expression basic?

=head1 SYNOPSIS

    use Bi::Visitor::IsBasic;
    
    if (Bi::Visitor::IsBasic->evaluate($expr)) {
        ...
    }

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::IsBasic;

use parent 'Bi::Visitor';
use warnings;
use strict;

use List::Util;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> Expression.

=back

Returns true if I<expr> evaluates to an identifier.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor; 
    bless $self, $class;
    
    while ($expr->isa('Bi::Expression::InlineIdentifier')) {
    	$expr = $expr->get_inline->get_expr;
    }
    my $arg = $expr->isa('Bi::Expression::VarIdentifier');
    
    #$expr->accept($self, \$arg);
    
    return $arg;
}

=item B<visit_after>(I<node>)

Visit node.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $arg = shift;
    
    if ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->get_expr->accept($self, $arg);
    } elsif (!$node->isa('Bi::Expression::VarIdentifier')) {
        $$arg = 0;
    }

    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
