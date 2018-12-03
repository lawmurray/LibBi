=head1 NAME

Bi::Expression::IntegerLiteral - integer literal.

=head1 SYNOPSIS

    use Bi::Expression::IntegerLiteral;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::IntegerLiteral;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<value>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $value = shift;

    assert (defined $value) if DEBUG;

    my $self = {
        _value => $value
    };
    bless $self, $class;
    
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = { %$self };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_value>

Get the literal value.

=cut
sub get_value {
    my $self = shift;
    return $self->{_value};
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
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_value == $obj->get_value); 
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

