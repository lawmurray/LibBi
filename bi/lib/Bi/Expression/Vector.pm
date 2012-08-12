=head1 NAME

Bi::Expression::Vector - vector of expressions.

=head1 SYNOPSIS

    use Bi::Expression::Vector;
    $A = new Bi::Expression::Matrix(4, 3);
    $b = new Bi::Expression::Vector(3);
    $c = new Bi::Expression::Vector(3);
    
    $c = $A*$b;  # basic operators are overloaded
    
    $expr = $c->get(0,0);
    $c->set(0, 0, $val);

=head1 INHERITS

L<Bi::Matrix>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Vector;

use base 'Bi::Expression::Matrix';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<size>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $size = shift;
    
    assert ($size >= 0) if DEBUG;

    my $self = new Bi::Expression::Matrix($size, 1);
    bless $self, $class;
    return $self;
}

=item B<size>

The size of the vector.

=cut
sub size {
    my $self = shift;
    return $self->num_rows
}

=item B<get>(I<i>)

Get an element.

=cut
sub get {
    my $self = shift;
    my $i = shift;

    return Bi::Expression::Matrix::get($self, $i, 0);
}

=item B<set>(I<i>, I<val>)

Set an element.

=cut
sub set {
    my $self = shift;
    my $i = shift;
    my $val = shift;

    Bi::Expression::Matrix::set($self, $i, 0, $val);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
