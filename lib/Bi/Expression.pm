=head1 NAME

Bi::Expression - arithmetic or conditional expression.

=head1 SYNOPSIS

    use Bi::Expression;

=head1 METHODS

=over 4

=cut

package Bi::Expression;

use parent 'Bi::Node';
use warnings;
use strict;
use overload
  '+' => \&_op_add,
  '-' => \&_op_sub,
  '*' => \&_op_mul,
  '/' => \&_op_div;

use Carp::Assert;

use Bi::Expression::Shape;
use Bi::Expression::BinaryOperator;
use Bi::Expression::ConstIdentifier;
use Bi::Expression::DimAliasIdentifier;
use Bi::Expression::Function;
use Bi::Expression::Index;
use Bi::Expression::InlineIdentifier;
use Bi::Expression::IntegerLiteral;
use Bi::Expression::Literal;
use Bi::Expression::Matrix;
use Bi::Expression::Range;
use Bi::Expression::StringLiteral;
use Bi::Expression::TernaryOperator;
use Bi::Expression::UnaryOperator;
use Bi::Expression::VarIdentifier;
use Bi::Expression::Vector;
use Bi::Visitor::GetNodesOfType;
use Bi::Visitor::EvalConst;
use Bi::Visitor::IsElement;
use Bi::Visitor::IsConst;
use Bi::Visitor::IsStatic;
use Bi::Visitor::IsBasic;
use Bi::Visitor::ToSymbolic;
use Bi::Visitor::Simplify;

=item B<is_const>

Is the expression constant? An expression is constant if it can be evaluated
to a constant value at compile time.

=cut
sub is_const {
    my $self = shift;

    return Bi::Visitor::IsConst->evaluate($self);
}

=item B<is_static>

Is the expression static? An expression is static if it is either constant,
or can be evaluated at runtime given only the values of parameters.

=cut
sub is_static {
    my $self = shift;
    
    return Bi::Visitor::IsStatic->evaluate($self);
}

=item B<is_common>

Is the expression common among multiple trajectories? An expression is common
if it is either static, or can be evaluated at runtime given only the values
of parameters and input variables.

=cut
sub is_common {
    my $self = shift;
    
    my $vars = $self->get_all_var_refs;
    my $var;
    my $type;
    
    foreach $var (@$vars) {
        $type = $var->get_var->get_type; 
        if ($type ne 'param' && $type ne 'param_aux_' && $type ne 'input' && $type ne 'obs') {
            return 0;
        }
    }
    return 1;
}

=item B<is_scalar>

Is this a scalar expression?

=cut
sub is_scalar {
    my $self = shift;
    
    return $self->get_shape->get_count == 0;
}

=item B<is_vector>

Is this a vector expression?

=cut
sub is_vector {
    my $self = shift;
    
    return $self->get_shape->get_count == 1;
}

=item B<is_matrix>

Is this a matrix expression?

=cut
sub is_matrix {
    my $self = shift;
    
    return $self->get_shape->get_count == 2;
}

=item B<is_element>

Is this an element expression?

=cut
sub is_element {
    my $self = shift;
    
    return Bi::Visitor::IsElement->evaluate($self);
}

=item B<is_basic>

Is this a basic expression?

=cut
sub is_basic {
    my $self = shift;

    return Bi::Visitor::IsBasic->evaluate($self);
}

=item B<is_zero>

Is this a constant expression that evaluates to zero?

=cut
sub is_zero {
    my $self = shift;
    
    return $self->is_const && $self->eval_const == 0.0;
}

=item B<is_one>

Is this a constant expression that evaluates to one?

=cut
sub is_one {
    my $self = shift;
    
    return $self->is_const && $self->eval_const == 1.0;
}

=item B<eval_const>

Evaluate the expression, if it is constant, and return the result. Undefined
if the expression is not constant.

=cut
sub eval_const {
    my $self = shift;
    
    return Bi::Visitor::EvalConst->evaluate($self);    
}

=item B<get_shape>

Get the shape of the result of the expression, as a L<Bi::Expression::Shape>
object.

=cut
sub get_shape {
    return new Bi::Expression::Shape();
}

=item B<d>(I<ident>)

Symbolically differentiate the expression with respect to I<ident>, and
return the new expression.

=cut
sub d {
    my $self = shift;
    my $ident = shift;
    
    my $symbolic = new Bi::Visitor::ToSymbolic;
    my $symb = $symbolic->expr2symb($self);
    my $resp = $symbolic->expr2symb($ident);
            
    my $d = Math::Symbolic::Operator->new('partial_derivative', $symb, $resp);
    my $dexpr = $symbolic->symb2expr($d->apply_derivatives->simplify);
        
    return $dexpr->simplify;
}

=item B<simplify>

Symbolically simplify the expression.

=cut
sub simplify {
    my $self = shift;
    return Bi::Visitor::Simplify->evaluate($self);
}

=item B<_op_binary>

Overloaded binary operator.

=cut
sub _op_binary {
    my $self = shift;
    my $other = shift;
    my $swap = shift;
    my $op = shift;

    my $expr;
    $self = _op_map($self);
    $other = _op_map($other);
    if ($swap) {
        $expr = new Bi::Expression::BinaryOperator($other, $op, $self);
    } else {
        $expr = new Bi::Expression::BinaryOperator($self, $op, $other);
    }
    return $expr->simplify;
}

sub _op_add {
    return _op_binary(@_, '+');
}

sub _op_sub {
    return _op_binary(@_, '-');
}

sub _op_mul {
    return _op_binary(@_, '*');
}

sub _op_div {
    return _op_binary(@_, '/');
}

=back

=head2 CLASS METHODS

=over 4

=item B<_op_map>(I<operand>)

Map operand for overloaded operator.

=cut
sub _op_map {
    my $operand = shift;
    
    if (!defined $operand) {
    	return new Bi::Expression::Literal(0);
    } elsif (ref($operand) && $operand->isa('Bi::Expression')) {
        return $operand;
    } elsif (ref($operand) eq 'STRING') {
        return new Bi::Expression::StringLiteral($operand);
    } else {
    	return new Bi::Expression::Literal($operand);
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

