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

use base 'Bi::Visitor';
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

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
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
