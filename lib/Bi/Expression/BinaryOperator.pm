=head1 NAME

Bi::Expression::BinaryOperator - binary operator and operands.

=head1 SYNOPSIS

    use Bi::Expression::BinaryOperator;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::BinaryOperator;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<expr1>, I<op>, I<expr2>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $expr1 = shift;
    my $op = shift;
    my $expr2 = shift;

    assert (defined $op) if DEBUG;
    assert (defined $expr1) if DEBUG;
    assert (defined $expr2) if DEBUG;

    my $self = {
        _expr1 => $expr1,
        _op => $op,
        _expr2 => $expr2
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
        _op => $self->get_op,
        _expr2 => $self->get_expr2->clone
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

=item B<get_expr1>

Get the left operand.

=cut
sub get_expr1 {
    my $self = shift;
    return $self->{_expr1};
}

=item B<get_expr2>

Get the right operand.

=cut
sub get_expr2 {
    my $self = shift;
    return $self->{_expr2};
}

=item B<get_shape>

Get the shape of the result of the expression, as a L<Bi::Expression::Shape>
object.

=cut
sub get_shape {
    my $self = shift;

    my $expr1 = $self->get_expr1;
    my $op = $self->get_op;
    my $expr2 = $self->get_expr2;

    if ($op eq '*' && ($expr1->is_vector || $expr1->is_matrix) && ($expr2->is_vector || $expr2->is_matrix)) {
    	if ($expr1->get_shape->get_size2 != $expr2->get_shape->get_size1) {
    		die("incompatible sizes in matrix/vector multiply\n");
    	}
    	my $shape = [];
    	if ($expr1->get_shape->get_size1 > 1) {
    		push(@$shape, $expr1->get_shape->get_size1);
    	}
    	if ($expr2->get_shape->get_size2 > 1) {
    		push(@$shape, $expr2->get_shape->get_size2);
    	}
        return new Bi::Expression::Shape($shape);
    } else {
        unless ($expr1->is_scalar || $expr2->is_scalar || $expr1->get_shape->equals($expr2->get_shape)) {
            die("incompatible sizes in element-wise operation\n");
        }
        if (!$self->get_expr1->is_scalar) {
	        return $self->get_expr1->get_shape;
        } else {
        	return $self->get_expr2->get_shape;
        }
    }
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
        $self->get_expr1->equals($obj->get_expr1) &&
        $self->get_expr2->equals($obj->get_expr2));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
