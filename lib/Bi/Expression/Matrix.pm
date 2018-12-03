=head1 NAME

Bi::Expression::Matrix - matrix of expressions.

=head1 SYNOPSIS

    use Bi::Expression::Matrix;
    $A = new Bi::Expression::Matrix(4, 3);
    $B = new Bi::Expression::Matrix(3, 4);
    $C = new Bi::Expression::Matrix(4, 4);
    
    $C = $A*$B;  # basic operators are overloaded
    
    $expr = $C->get(0,0);
    $C->set(0, 0, $val);

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Matrix;

use parent 'Bi::Expression::MatrixReference';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<rows>, I<cols>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $rows = shift;
    my $cols = shift;
    
    assert ($rows >= 0) if DEBUG;
    assert ($cols >= 0) if DEBUG;

    my $self = new Bi::Expression::MatrixReference([], 0, $rows, 0, $cols, $rows);
    bless $self, $class;

    $self->clear;
    
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
        _data => [],
        _start1 => $self->{_start1},
        _size1 => $self->{_size1},
        _start2 => $self->{_start2},
        _size2 => $self->{_size2},
        _lead => $self->{_lead}
    };
    bless $clone, ref($self);
    
    # copy data too
    for (my $j = 0; $j < $clone->size2; ++$j) {
        for (my $i = 0; $i < $clone->size1; ++$i) {
            $clone->set($i, $j, $self->get($i, $j)->clone);
        }
    }
    return $clone;
}

=item B<swap>

Swap contents of matrix with another matrix.

=cut
sub swap {
    my $self = shift;
    my $other = shift;
    
    assert ($other->isa('Bi::Expression::Matrix')) if DEBUG;
    
    my $self_data = $self->get_data;
    my $other_data = $other->get_data;
    
    $self->{_data} = $other_data;
    $other->{_data} = $self_data;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

