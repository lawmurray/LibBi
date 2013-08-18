=head1 NAME

Bi::Expression::Shape - shape of an expression.

=head1 SYNOPSIS

    use Bi::Expression::Shape;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Shape;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<sizes>)

Constructor.

=over 4

=item I<sizes>

Array ref giving the sizes of the result of an expression along each
dimension. Empty for scalar.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $sizes = shift;
    
    assert (!defined $sizes || ref($sizes) eq 'ARRAY') if DEBUG;
    
    if (!defined $sizes) {
        $sizes = [];
    }
    
    my $self = {
        _sizes => $sizes
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
    	_sizes => [ @{$self->get_sizes} ]
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_sizes>

Get the starting index.

=cut
sub get_sizes {
    my $self = shift;
    return $self->{_sizes};
}

=item B<get_count>

Get the number of dimensions.

=cut
sub get_count {
    my $self = shift;
    return scalar(@{$self->get_sizes});
}

=item B<get_size1>

Get the virtual number of rows.

=cut
sub get_size1 {
    my $self = shift;
    if ($self->get_count >= 1) {
        return $self->get_sizes->[0];
    } else {
        return 1;
    }
}

=item B<get_size2>

Get the virtual number of columns.

=cut
sub get_size2 {
    my $self = shift;
    if ($self->get_count >= 2) {
        return $self->get_sizes->[1];
    } else {
        return 1;
    }
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    my $equals = $self->get_count == $obj->get_count;
    for (my $i = 0; $equals && $i < $self->get_count; ++$i) {
        $equals = $equals && $self->get_sizes->[$i] == $obj->get_sizes->[$i];
    }
    return $equals;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
