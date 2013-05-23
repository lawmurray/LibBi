=head1 NAME

Bi::Expression::TernaryOperator - ternary operator and its operands.

=head1 SYNOPSIS

    use Bi::Expression::TernaryOperator

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::TernaryOperator;

use parent 'Bi::Expression';
use warnings;
use strict;

use Bi::Visitor::ToSymbolic;

use Carp::Assert;

=item B<new>(I<expr1>, I<op1>, I<expr2>, I<op2>, I<expr3>)

Constructor.

=over 4

=item I<expr1>

First operand.

=item I<op1>

First operator.

=item I<expr2>

Second operand.

=item I<op2>

Second operator.

=item I<expr3>

Third operand.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $expr1 = shift;
    my $op1 = shift;
    my $expr2 = shift;
    my $op2 = shift;
    my $expr3 = shift;
    
    assert (defined $expr1 && $expr1->isa('Bi::Expression')) if DEBUG;
    assert (defined $op1) if DEBUG;
    assert (defined $expr2 && $expr2->isa('Bi::Expression')) if DEBUG;
    assert (defined $op2) if DEBUG;
    assert (defined $expr3 && $expr3->isa('Bi::Expression')) if DEBUG;

    my $self = {
        _expr1 => $expr1,
        _op1 => $op1,
        _expr2 => $expr2,
        _op2 => $op2,
        _expr3 => $expr3
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
        _expr1 => $self->get_expr1->clone,
        _op1 => $self->get_op1,
        _expr2 => $self->get_expr2->clone,
        _op2 => $self->get_op2,
        _expr3 => $self->get_expr3->clone
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_expr1>

Get the first operand.

=cut
sub get_expr1 {
    my $self = shift;
    return $self->{_expr1};
}

=item B<get_op1>

Get the first operator.

=cut
sub get_op1 {
    my $self = shift;
    return $self->{_op1};
}

=item B<set_op1>(I<op>)

Set the first operator.

=cut
sub set_op1 {
    my $self = shift;
    my $op = shift;
    
    $self->{_op1} = $op;
}

=item B<get_expr2>

Get the second operand.

=cut
sub get_expr2 {
    my $self = shift;
    return $self->{_expr2};
}

=item B<get_op2>

Get the second operator.

=cut
sub get_op2 {
    my $self = shift;
    return $self->{_op2};
}

=item B<set_op2>(I<op>)

Set the second operator.

=cut
sub set_op2 {
    my $self = shift;
    my $op = shift;
    
    $self->{_op2} = $op;
}

=item B<get_expr3>

Get the third operand.

=cut
sub get_expr3 {
    my $self = shift;
    return $self->{_expr3};
}

=item B<d>(I<ident>)

Symbolically differentiate the expression with respect to I<ident>, and
return the new expression. For an expression with the ternary operator at
the top level, this applies the same condition, replacing the true and
false clauses with their derivatives.

=cut
sub d {
    my $self = shift;
    my $ident = shift;
    
    my $symbolic = new Bi::Visitor::ToSymbolic;
    my $symb2 = $symbolic->expr2symb($self->get_expr2);
    my $symb3 = $symbolic->expr2symb($self->get_expr3);
    my $resp = $symbolic->expr2symb($ident);
            
    my $d2 = Math::Symbolic::Operator->new('partial_derivative', $symb2, $resp);
    my $d3 = Math::Symbolic::Operator->new('partial_derivative', $symb3, $resp);
    my $dexpr2 = $symbolic->symb2expr($d2->apply_derivatives->simplify)->simplify;
    my $dexpr3 = $symbolic->symb2expr($d3->apply_derivatives->simplify)->simplify;

    return new Bi::Expression::TernaryOperator($self->get_expr1->clone,
        $self->get_op1, $dexpr2, $self->get_op2, $dexpr3);
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;
    
    $self = $visitor->visit_before($self, @args);
    $self->{_expr1} = $self->get_expr1->accept($visitor, @args);
    $self->{_expr2} = $self->get_expr2->accept($visitor, @args);
    $self->{_expr3} = $self->get_expr3->accept($visitor, @args);

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
        $self->get_expr1->equals($obj->get_expr1) &&
        $self->get_op1 eq $obj->get_op1 &&
        $self->get_expr2->equals($obj->get_expr2) &&
        $self->get_op2 eq $obj->get_op2 &&
        $self->get_expr3->equals($obj->get_expr3));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
