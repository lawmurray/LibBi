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

use base 'Bi::ArgHandler', 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
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
    $self->{_name} = $name;
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

=item B<is_math>

Is the function a math function?

=cut
sub is_math {
    my $self = shift;    
    my %MATH_FUNCTIONS = (
      'fabs' => 1,
      'log' => 1,
      'exp' => 1,
      'max' => 1,
      'min' => 1,
      'sqrt' => 1,
      'pow' => 1,
      'fmod' => 1,
      'modf' => 1,
      'ceil' => 1,
      'floor' => 1,
      'tgamma' => 1,
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
      'atanh' => 1
    );
    
    return exists $MATH_FUNCTIONS{$self->get_name};
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    Bi::ArgHandler::accept($self, $visitor, @args);

    return $visitor->visit($self, @args);
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    my $equals = 
        ref($obj) eq ref($self) &&
        $self->get_name eq $obj->get_name &&
        $self->Bi::ArgHandler::equals($obj);
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
