=head1 NAME

Bi::Expression::UnaryOperator - unary operator and its operand.

=head1 SYNOPSIS

    use Bi::Expression::UnaryOperator;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::UnaryOperator;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<op>, I<expr>)

Constructor.

=over 4

=item I<op> Operator.

=item I<expr> Operand.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $op = shift;
    my $expr = shift;

    assert (defined $op) if DEBUG;
    assert (defined $expr) if DEBUG;
    
    my $self = {
        _op => $op,
        _expr => $expr
    };
    bless $self, $class;
    
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
        _op => $self->get_op,
        _expr => $self->get_expr->clone
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_op>

Get the operator.

=cut
sub get_op {
    my $self = shift;
    return $self->{_op};
}

=item B<set_op>(I<op>)

Set the operator.

=cut
sub set_op {
    my $self = shift;
    my $op = shift;
    
    $self->{_op} = $op;
}

=item B<get_expr>

Get the operand.

=cut
sub get_expr {
    my $self = shift;
    return $self->{_expr};
}

=item B<get_shape>

Get the shape of the result of the expression, as a L<Bi::Expression::Shape>
object.

=cut
sub get_shape {
    my $self = shift;

    return $self->get_expr->get_shape;
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    $self = $visitor->visit_before($self, @args);
    $self->{_expr} = $self->get_expr->accept($visitor, @args);

    return $visitor->visit_after($self, @args);
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_op eq $obj->get_op &&
        $self->get_expr->equals($obj->get_expr));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
