=head1 NAME

Bi::Expression::Vector - vector of expressions.

=head1 INHERITS

L<Bi::Matrix>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Vector;

use parent 'Bi::Expression::VectorReference';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<size>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $size = shift;
    
    assert ($size >= 0) if DEBUG;

    my $self = new Bi::Expression::VectorReference([], 0, $size);    
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
    for (my $i = 0; $i < $clone->size1; ++$i) {
        $clone->set($i, $self->get($i)->clone);
    }

    return $clone;
}

=item B<swap>

Swap contents of vector with another vector.

=cut
sub swap {
    my $self = shift;
    my $other = shift;
    
    assert ($other->isa('Bi::Expression::Vector')) if DEBUG;
    
    my $self_data = $self->get_data;
    my $other_data = $other->get_data;
    
    $self->{_data} = $other_data;
    $other->{_data} = $self_data;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

