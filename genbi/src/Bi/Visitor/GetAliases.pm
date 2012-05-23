=head1 NAME

Bi::Visitor::GetAliases - visitor for constructing list of dimension
aliases used in an expression.

=head1 SYNOPSIS

    use Bi::Visitor::GetAliases;
    $vars = Bi::Visitor::GetAliases->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::GetAliases;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns an array ref containing all the unique dimension aliases in the
expression, as strings.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor; 
    $self->{_aliases} = [];
    bless $self, $class;
    
    $expr->accept($self);
    
    return $self->{_aliases};
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    
    if ($node->isa('Bi::Expression::Offset')) {
        Bi::Utility::push_unique($self->{_aliases}, $node->get_alias);
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
