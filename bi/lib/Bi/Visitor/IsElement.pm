=head1 NAME

Bi::Visitor::IsElement - is result of expression an element?

=head1 SYNOPSIS

    use Bi::Visitor::IsElement;
    
    if (Bi::Visitor::IsElement->evaluate($expr)) {
        ...
    }

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::IsElement;

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
    
    my $args = [];
    $expr->accept($self, $args);
    
    return pop @$args;
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $args = shift;
    my $is = 0;
    
    if ($node->isa('Bi::Expression::Index')) {
        $is = 1;
    } else {
        my $num_args = 0;
        if ($node->isa('Bi::Expression::BinaryOperator')) {
            $num_args = 2;
        } elsif ($node->isa('Bi::Expression::Function')) {
            $num_args = $node->num_args + $node->num_named_args;
        } elsif ($node->isa('Bi::Expression::TernaryOperator')) {
            $num_args = 3;
        } elsif ($node->isa('Bi::Expression::UnaryOperator')) {
            $num_args = 1;
        } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
            $num_args = $node->num_indexes;
        }

        if ($num_args) {
            my @ch = splice(@$args, -$num_args);
            if (@ch) {
                $is = List::Util::max(@ch);
            }
        }
    }
    
    push(@$args, $is);
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
