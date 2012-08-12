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

use base 'Bi::Expression';
use warnings;
use strict;
use overload
    '+' => \&_op_add,
    '-' => \&_op_sub,
    '*' => \&_op_mul,
    '/' => \&_op_div;

use Carp::Assert;

=item B<new>(I<rows>, I<cols>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $rows = shift;
    my $cols = shift;
    
    assert ($rows >= 0) if DEBUG;
    assert ($cols >= 0) if DEBUG;

    my $self = {
        _rows => $rows,
        _cols => $cols,
        _data => []
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
        _rows => $self->{_rows},
        _cols => $self->{_cols},
        _data => []
    };
    bless $clone, ref($self);

    my $size = $self->num_rows*$self->num_cols;
    for (my $i = 0; $i < $size; ++$i) {
        if (defined $self->{_data}->[$i]) {
            $clone->{_data}->[$i] = $self->{_data}->[$i]->clone;
        }
    }
    
    return $clone;
}

=item B<num_rows>

The number of rows.

=cut
sub num_rows {
    my $self = shift;
    return $self->{_rows};
}

=item B<num_cols>

The number of columns.

=cut
sub num_cols {
    my $self = shift;
    return $self->{_cols};
}

=item B<get_data>

The underlying array of data.

=cut
sub get_data {
    my $self = shift;
    return $self->{_data};
}

=item B<get>(I<row>, I<col>)

Get an element.

=cut
sub get {
    my $self = shift;
    my $row = shift;
    my $col = shift;
    
    assert ($row >= 0 && $row < $self->num_rows) if DEBUG;
    assert ($col >= 0 && $col < $self->num_cols) if DEBUG;
    
    return $self->get_data->[$col*$self->num_rows + $row];
}

=item B<set>(I<row>, I<col>, I<val>)

Set an element.

=cut
sub set {
    my $self = shift;
    my $row = shift;
    my $col = shift;
    my $val = shift;

    assert ($row >= 0 && $row < $self->num_rows) if DEBUG;
    assert ($col >= 0 && $col < $self->num_cols) if DEBUG;

    $self->get_data->[$col*$self->num_rows + $row] = $val;    
}

=item B<clear>

Clear the matrix.

=cut
sub clear {
    my $self = shift;
    $self->{_data} = [];
}

=item <is_empty>

Is the matrix empty?

=cut
sub is_empty {
    my $self = shift;
    my $n = $self->num_rows*$self->num_cols;
    
    for (my $i = 0; $i < $n; ++$i) {
        if (defined $self->get_data->[$i]) {
           return 0;
        }
    }
    return 1;
}

=item B<ident>

Set the leading diagonal to all ones.

=cut
sub ident {
    my $self = shift;
    $self->clear;
    
    for (my $i = 0; $i < $self->num_rows && $i < $self->num_cols; ++$i) {
        $self->set($i, $i, new Bi::Expression::Literal(1));
    }
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
    $self->{_other} = $self_data;
}

=item B<_op_binary>

Overloaded + operator.

=cut
sub _op_binary {
    my $self = shift;
    my $other = shift;
    my $swap = shift;
    my $op = shift;

    my $result = new Bi::Expression::Matrix($self->num_rows, $self->num_cols);
    for (my $j = 0; $j < $self->num_cols; ++$j) {
        for (my $i = 0; $i < $self->num_rows; ++$i) {
            my $self1 = $self->get($i, $j);
            my $other1 = $other->get($i, $j);
            
            if (!defined $self1) {
                $self1 = 0;
            }
            if (!defined $other1) {
                $other1 = 0;
            }
            $result->set($i, $j, Bi::Expression::_op_binary($self1, $other1, $swap, $op));
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
    my ($A, $B, $C);
    if ($swap) {
        $A = $other;
        $B = $self;
    } else {
        $A = $self;
        $B = $other;
    }
    assert ($A->num_cols == $B->num_rows) if DEBUG;
    $C = new Bi::Expression::Matrix($A->num_rows, $B->num_cols);
    
    # multiply
    my ($i, $j, $k, $c);
    for ($k = 0; $k < $B->num_cols; ++$k) {
        for ($i = 0; $i < $A->num_rows; ++$i) {
            $c = $C->get($i, $k);
            for ($j = 0; $j < $A->num_cols; ++$j) {
                my $a = $A->get($i, $j);
                my $b = $B->get($j, $k);
                if (defined $a && defined $b) { # otherwise result just zero
                    if (defined $c) { 
                        $c += $a*$b;
                    } else {
                        $c = $a*$b;
                    }                    
                }
            }
            if (defined $c) {
                $c = $c->simplify;
            }
            $C->set($i, $k, $c);
        }    
    }
    
    return $C;
}

sub _op_div {
    assert(0) if DEBUG;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
