=head1 NAME

Bi::Expression::MatrixReference - matrix of expressions.

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::MatrixReference;

use parent 'Bi::Expression';
use warnings;
use strict;
use overload
    '+' => \&_op_add,
    '-' => \&_op_sub,
    '*' => \&_op_mul;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<data>, I<start1>, I<size1>, I<start2>, I<size2>, I<lead>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $data = shift;
    my $start1 = shift;
    my $size1 = shift;
    my $start2 = shift;
    my $size2 = shift;
    my $lead = shift;
    
    assert ($start1 >= 0) if DEBUG;
    assert ($size1 >= 0) if DEBUG;
    assert ($start2 >= 0) if DEBUG;
    assert ($size2 >= 0) if DEBUG;
    assert ($lead >= $start1 + $size1) if DEBUG;

    my $self = {
        _data => $data,
        _start1 => $start1,
        _size1 => $size1,
        _start2 => $start2,
        _size2 => $size2,
        _lead => $lead
    };
    bless $self, $class;
    
    return $self;
}

=item B<data>

Underlying data.

=cut
sub data {
    my $self = shift;
    return $self->{_data};
}

=item B<start1>

Starting row in underlying data.

=cut
sub start1 {
    my $self = shift;
    return $self->{_start1};
}

=item B<size1>

The number of rows.

=cut
sub size1 {
    my $self = shift;
    return $self->{_size1};
}

=item B<start2>

Starting column in underlying data.

=cut
sub start2 {
    my $self = shift;
    return $self->{_start2};
}

=item B<size2>

The number of columns.

=cut
sub size2 {
    my $self = shift;
    return $self->{_size2};
}

=item B<lead>

=cut
sub lead {
    my $self = shift;
    return $self->{_lead};
}

=item B<get>(I<row>, I<col>)

Get an element.

=cut
sub get {
    my $self = shift;
    my $row = shift;
    my $col = shift;
    
    assert ($row >= 0 && $row < $self->size1) if DEBUG;
    assert ($col >= 0 && $col < $self->size2) if DEBUG;
    
    return $self->data->[($self->start2 + $col)*$self->lead + $self->start1 + $row];
}

=item B<set>(I<row>, I<col>, I<val>)

Set an element.

=cut
sub set {
    my $self = shift;
    my $row = shift;
    my $col = shift;
    my $val = shift;

    assert ($row >= 0 && $row < $self->size1) if DEBUG;
    assert ($col >= 0 && $col < $self->size2) if DEBUG;
    assert (ref($val) && $val->isa('Bi::Expression')) if DEBUG;

    $self->data->[($self->start2 + $col)*$self->lead + $self->start1 + $row] = $val;
}

=item B<clear>

Clear the matrix.

=cut
sub clear {
    my $self = shift;

    for (my $j = 0; $j < $self->size2; ++$j) {
        for (my $i = 0; $i < $self->size1; ++$i) {
            $self->set($i, $j, new Bi::Expression::Literal(0.0));
        }
    }
}

=item B<ident>

Set the leading diagonal to all ones.

=cut
sub ident {
    my $self = shift;
    
    for (my $j = 0; $j < $self->size2; ++$j) {
        for (my $i = 0; $i < $self->size1; ++$i) {
            if ($i == $j) {
                $self->set($i, $j, new Bi::Expression::Literal(1.0));
            } else {
                $self->set($i, $j, new Bi::Expression::Literal(0.0));
            }
        }
    }
}

=item B<subrange>(I<start1>, I<size1>, I<start2>, I<size2>)

Get a subrange of the matrix.

=cut
sub subrange {
    my $self = shift;
    my $start1 = shift;
    my $size1 = shift;
    my $start2 = shift;
    my $size2 = shift;
    
    assert ($start1 >= 0 && $start1 + $size1 <= $self->size1) if DEBUG;
    assert ($start2 >= 0 && $start2 + $size2 <= $self->size2) if DEBUG;
    
    return new Bi::Expression::MatrixReference($self->data,
            $self->start1 + $start1, $size1, $self->start2 + $start2, $size2,
            $self->lead);
}

=item B<assign>(I<other>)

Assignment.

=cut
sub assign {
    my $self = shift;
    my $other = shift;

    assert($other->isa('Bi::Expression::MatrixReference')) if DEBUG;
    assert($self->size1 == $other->size1) if DEBUG;
    assert($self->size2 == $other->size2) if DEBUG;

    for (my $j = 0; $j < $self->size2; ++$j) {
        for (my $i = 0; $i < $self->size1; ++$i) {
            $self->set($i, $j, $other->get($i, $j)->clone);
        }
    }
}

=item B<_op_binary>

Overloaded + operator.

=cut
sub _op_binary {
    my $self = shift;
    my $other = shift;
    my $swap = shift;
    my $op = shift;

    my $result = new Bi::Expression::Matrix($self->size1, $self->size2);
    for (my $j = 0; $j < $self->size2; ++$j) {
        for (my $i = 0; $i < $self->size1; ++$i) {
            $result->set($i, $j, Bi::Expression::_op_binary($self->get($i, $j), $other->get($i, $j), $swap, $op));
        }
    }
    return $result;
}

sub _op_add {
    return _op_binary(@_, '+');
}

sub _op_sub {
    return _op_binary(@_, '-');
}

sub _op_mul {
    my $self = shift;
    my $other = shift;
    my $swap = shift;
    my $op = shift;
    
    # set up operands and result
    my ($A, $B);
    if ($swap) {
        $A = $other;
        $B = $self;
    } else {
        $A = $self;
        $B = $other;
    }
    assert ($A->size2 == $B->size1) if DEBUG;
    my $C = new Bi::Expression::Matrix($A->size1, $B->size2);
    
    # multiply
    my ($i, $j, $k, $c);
    for ($k = 0; $k < $B->size2; ++$k) {
        for ($i = 0; $i < $A->size1; ++$i) {
            $c = $C->get($i, $k);
            for ($j = 0; $j < $A->size2; ++$j) {
                my $a = $A->get($i, $j);
                my $b = $B->get($j, $k);
                $c += $a*$b;
            }
            $c = $c->simplify;
            $C->set($i, $k, $c);
        }    
    }
    
    return $C;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 3742 $ $Date: 2013-04-17 19:31:57 +1000 (Wed, 17 Apr 2013) $
