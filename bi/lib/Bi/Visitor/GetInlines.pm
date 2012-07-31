=head1 NAME

Bi::Visitor::GetInlines - visitor for constructing list of inlines
referenced by an expression.

=head1 SYNOPSIS

    use Bi::Visitor::GetInlines;
    $inlines = Bi::Visitor::GetInlines->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::GetInlines;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns an array ref containing all the unique
L<Bi::Expression::InlineIdentifier> objects in the expression.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor; 
    $self->{_inlines} = [];
    bless $self, $class;
    
    $expr->accept($self);
    
    return $self->{_inlines};
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    
    if ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->accept($self);
        Bi::Utility::push_unique($self->{_inlines}, $node);
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
