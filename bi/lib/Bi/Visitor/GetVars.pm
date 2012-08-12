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

use Bi::Utility qw(contains push_unique);

=item B<evaluate>(I<expr>)

Evaluate.

=over 4

=item I<expr> L<Bi::Expression> object.

=item I<types>  (optional) Types of variables to return. If I<types> is given
as a string, only variables of that type are returned. If I<types> is given
as an array ref of strings, only variables of those types are returned.

=back

Returns an array ref containing all the unique
L<Bi::Expression::VarIdentifier> objects in the expression of the given
types. If I<types> is not given, returns all variables, regardless of type.

=cut
sub evaluate {
    my $class = shift;
    my $expr = shift;
    my $types = [];
    if (@_) {
      $types = shift;
    }
    if (ref($types) ne 'ARRAY') {
        $types = [ $types ];
    }
    
    my $self = new Bi::Visitor; 
    bless $self, $class;

    my $vars = [];
    $expr->accept($self, $types, $vars);
    
    return $vars;
}

=item B<visit>(I<node>)

Visit node.

=cut
sub visit {
    my $self = shift;
    my $node = shift;
    my $types = shift;
    my $vars = shift;
    
    if ($node->isa('Bi::Expression::VarIdentifier')) {
        my $include = !@$types;
        foreach my $type (@$types) {
            if ($type eq $node->get_var->get_type) {
                $include = 1;
                last;
            }
        }
        if ($include) {
            push_unique($vars, $node);
        }
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $node->get_inline->get_expr->accept($self, $types, $vars);
    }

    return $node;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
