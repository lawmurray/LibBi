=head1 NAME

Bi::Visitor::GetConsts - visitor for constructing list of constants
referenced by an expression.

=head1 SYNOPSIS

    use Bi::Visitor::GetConsts;
    $consts = Bi::Visitor::GetConsts->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::GetConsts;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns an array ref containing all the unique
L<Bi::Expression::ConstIdentifier> objects in the expression.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    
    my $self = new Bi::Visitor; 
    $self->{_consts} = [];
    bless $self, $class;
    
    $expr->accept($self);
    
    return $self->{_consts};
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    
    if ($node->isa('Bi::Expression::ConstIdentifier')) {
        Bi::Utility::push_unique($self->{_consts}, $node);
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->accept($self);
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
