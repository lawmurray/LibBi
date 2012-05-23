=head1 NAME

Bi::Visitor::GetVars - visitor for constructing list of variables
referenced by an expression.

=head1 SYNOPSIS

    use Bi::Visitor::GetVars;
    $vars = Bi::Visitor::GetVars->evaluate($expr);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::GetVars;

use base 'Bi::Visitor';
use warnings;
use strict;

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=item I<type>  (optional) Type of variables.

=back

Returns an array ref containing all the unique
L<Bi::Expression::VarIdentifier> objects in the expression of the given type.
If I<type> is not given, returns all variables, regardless of type.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    my $type = shift;
    
    my $self = new Bi::Visitor; 
    $self->{_vars} = [];
    bless $self, $class;
    
    $expr->accept($self, $type);
    
    return $self->{_vars};
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $type = shift;
    
    if ($node->isa('Bi::Expression::VarIdentifier')) {
        if (!defined($type) || $node->get_var->get_type eq $type) {
            Bi::Utility::push_unique($self->{_vars}, $node);
        }
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->accept($self, $type);
    }
    
    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
