=head1 NAME

Bi::Expression - arithmetic or conditional expression.

=head1 SYNOPSIS

    use Bi::Expression;

=head1 METHODS

=over 4

=cut

package Bi::Expression;

use warnings;
use strict;

use Carp::Assert;

use Bi::Expression::BinaryOperator;
use Bi::Expression::Function;
use Bi::Expression::ConstIdentifier;
use Bi::Expression::InlineIdentifier;
use Bi::Expression::VarIdentifier;
use Bi::Expression::Literal;
use Bi::Expression::Offset;
use Bi::Expression::DimAlias;
use Bi::Expression::Parens;
use Bi::Expression::StringLiteral;
use Bi::Expression::TernaryOperator;
use Bi::Expression::UnaryOperator;

use Bi::Visitor::GetConsts;
use Bi::Visitor::GetInlines;
use Bi::Visitor::GetVars;
use Bi::Visitor::GetDims;
use Bi::Visitor::GetAliases;
use Bi::Visitor::EvalConst;
use Bi::Visitor::IsElement;
use Bi::Visitor::IsConst;
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
    
    my $vars = $self->get_vars;
    my $var;
    my $type;
    
    foreach $var (@$vars) {
        $type = $var->get_var->get_type; 
        if ($type ne 'param' && $type ne 'param_aux_') {
            return 0;
        }
    }
    return 1;
}

=item B<is_common>

Is the expression common among multiple trajectories? An expression is common
if it is either static, or can be evaluated at runtime given only the values
of parameters and input variables.

=cut
sub is_common {
    my $self = shift;
    
    my $vars = $self->get_vars;
    my $var;
    my $type;
    
    foreach $var (@$vars) {
        $type = $var->get_var->get_type; 
        if ($type ne 'param' && $type ne 'param_aux_' && $type ne 'input') {
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
    
    return $self->num_dims == 0;
}

=item B<is_vector>

Is this a vector expression?

=cut
sub is_vector {
    my $self = shift;
    
    return $self->num_dims == 1;
}

=item B<is_matrix>

Is this a matrix expression?

=cut
sub is_matrix {
    my $self = shift;
    
    return $self->num_dims == 2;
}

=item B<is_element>

Is this an element expression?

=cut
sub is_element {
    my $self = shift;
    
    return Bi::Visitor::IsElement->evaluate($self);
}

=item B<eval_const>

Evaluate the expression, if it is constant, and return the result. Undefined
if the expression is not constant.

=cut
sub eval_const {
    my $self = shift;
    
    return Bi::Visitor::EvalConst->evaluate($self);    
}

=item B<num_consts>

Number of constants referenced in the expression.

=cut
sub num_consts {
    my $self = shift;
    
    return scalar(@{Bi::Visitor::GetConsts->evaluate($self)});
}

=item B<get_consts>

Get all constants referenced in the expression, as a list of
L<Bi::Expression::ConstIdentifier> objects.

=cut
sub get_consts {
    my $self = shift;
    
    return Bi::Visitor::GetConsts->evaluate($self);
}

=item B<num_inlines>

Number of inlines referenced in the expression.

=cut
sub num_inlines {
    my $self = shift;
    
    return scalar(@{Bi::Visitor::GetInlines->evaluate($self)});
}

=item B<get_inlines>

Get all inlines referenced in the expression, as a list of
L<Bi::Expression::InlineIdentifier> objects.

=cut
sub get_inlines {
    my $self = shift;
    
    return Bi::Visitor::GetInlines->evaluate($self);
}

=item B<num_vars>

Number of variables referenced in the expression.

=cut
sub num_vars {
    my $self = shift;

    return scalar(@{Bi::Visitor::GetVars->evaluate($self)});
}

=item B<get_vars>(I<type>)

Get all variables referenced in the expression, as a list of
L<Bi::Expression::VarIdentifier> objects. If I<type> is given, returns only
variables of that type, otherwise returns all variables.

=cut
sub get_vars {
    my $self = shift;
    my $type;
    if (@_) {
        $type = shift;
    }

    return Bi::Visitor::GetVars->evaluate($self, $type);
}

=item B<num_dims>

Number of dimensions along which the result of the expression is defined.

=cut
sub num_dims {
    my $self = shift;
    
    return scalar(@{$self->get_dims});
}

=item B<get_dims>

Get all dimensions along which the result of the expression extends, as a
list of L<Bi::Expression::Dim> objects.

=cut
sub get_dims {
    my $self = shift;
    
    return Bi::Visitor::GetDims->evaluate($self);
}

=item B<num_aliases>

Number of dimension aliases used in the expression.

=cut
sub num_aliases {
    my $self = shift;
    
    return scalar(@{Bi::Visitor::GetAliases->evaluate($self)});
}

=item B<get_aliases>

Get all dimension aliases used in the expression, as strings.

=cut
sub get_aliases {
    my $self = shift;
    
    return Bi::Visitor::GetAliases->evaluate($self);
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
    
    return $dexpr;
}

=item B<simplify>

Symbolically simplify the expression.

=cut
sub simplify {
    my $self = shift;
    Bi::Visitor::Simplify->evaluate($self);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
