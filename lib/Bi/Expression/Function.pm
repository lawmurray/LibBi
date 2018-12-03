=head1 NAME

Bi::Expression::Function - function call.

=head1 SYNOPSIS

    use Bi::Expression::Function;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Function;

use parent 'Bi::ArgHandler', 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

use Bi::Utility;

=item B<new>(I<name>, I<args>, I<named_args>)

Constructor.

=over 4

=item I<name>

Name of the function.

=item I<args>

Ordered list of positional arguments to the function as L<Bi::Expression>
objects.

=item I<named_args>

Hash of named arguments to the function, keyed by name, as L<Bi::Expression>
objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $args = shift;
    my $named_args = shift;

    my $self = new Bi::ArgHandler($args, $named_args);
    my $shape = determine_shape($self, $name);
    $self->{_name} = $name;
    $self->{_shape} = $shape;
    bless $self, $class;
   
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = Bi::ArgHandler::clone($self);
    $clone->{_name} = $self->get_name;
    $clone->{_shape} = $self->get_shape->clone;
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_name>

Get the name of the function.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<is_action>

Is the function a named action? This simply checks the function's name
against the list of all actions. Some uses can only be distinguished
by context (e.g. C<gamma>).

=cut
sub is_action {
    my $self = shift;
    
    my $name = $self->get_name;
    my $class = "Bi::Action::$name";
    (eval ("require $class") && return 1) || return 0;
}

=item B<is_math>

Is the function a math function? This simply checks the function's name
against the list of all math functions. Some uses can only be distinguished
by context (e.g. C<gamma>).

=cut
sub is_math {
    my $self = shift;    
    my %MATH_FUNCTIONS = (
      'abs' => 1,
      'log' => 1,
      'exp' => 1,
      'max' => 1,
      'min' => 1,
      'sqrt' => 1,
      'rsqrt' => 1,
      'pow' => 1,
      'mod' => 1,
      'ceil' => 1,
      'floor' => 1,
      'round' => 1,
      'gamma' => 1,
      'lgamma' => 1,
      'sin' => 1,
      'cos' => 1,
      'tan' => 1,
      'asin' => 1,
      'acos' => 1,
      'atan' => 1,
      'atan2' => 1,
      'sinh' => 1,
      'cosh' => 1,
      'tanh' => 1,
      'asinh' => 1,
      'acosh' => 1,
      'atanh' => 1,
      'erf', => 1,
      'erfc', => 1
    );
    
    return exists $MATH_FUNCTIONS{$self->get_name};
}

=item B<determine_shape>

Determine the shape of the result of the expression, as a L<Bi::Expression::Shape>
object.

=cut
sub determine_shape {
  my $self = shift;
  my $name = shift;
	if ($name eq 'gemv_') {
		my $expr1 = $self->get_arg(0);
		my $expr2 = $self->get_arg(1);
		
		if ($expr1->get_shape->get_size2 != $expr2->get_shape->get_size1) {
    		die("incompatible sizes in matrix/vector multiply\n");
    	}
    	my $sizes = [ $expr1->get_shape->get_size1 ];
        return new Bi::Expression::Shape($sizes);
	} elsif ($name eq 'gemm_') {
		my $expr1 = $self->get_arg(0);
		my $expr2 = $self->get_arg(1);
		
		if ($expr1->get_shape->get_size2 != $expr2->get_shape->get_size1) {
    		die("incompatible sizes in matrix/vector multiply\n");
    	}
    	my $sizes = [ $expr1->get_shape->get_size1, $expr2->get_shape->get_size2 ];
        return new Bi::Expression::Shape($sizes);
	} elsif ($name eq 'transpose') {
		my $expr1 = $self->get_arg(0);
    	my $shape = [];
    	push(@$shape, $expr1->get_shape->get_size2);
    	if ($expr1->get_shape->get_size1 > 1) {
    		push(@$shape, $expr1->get_shape->get_size1);
    	}
        return new Bi::Expression::Shape($shape);
    } elsif ($self->num_args > 0) {
    	# assume return value is same shape as first argument
    	return $self->get_arg(0)->get_shape->clone;
    } else {
    	# scalar
        return new Bi::Expression::Shape();
    }
}

=item B<get_shape>

Get the shape of the result of the expression, as a L<Bi::Expression::Shape>
object.

=cut
sub get_shape {
    my $self = shift;

    return $self->{_shape};
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    my $new = $visitor->visit_before($self, @args);
    if (refaddr($new) == refaddr($self)) {
	    Bi::ArgHandler::accept($self, $visitor, @args);
	    $new = $visitor->visit_after($self, @args);
    }
    return $new;
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (ref($obj) eq ref($self) &&
        $self->get_name eq $obj->get_name &&
        $self->Bi::ArgHandler::equals($obj));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

