=head1 NAME

Bi::Expression::VarIdentifier - reference to variable, with or without
dimension offsets.

=head1 SYNOPSIS

    use Bi::Expression::VarIdentifier;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::VarIdentifier;

use base 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<var>, I<offsets>, I<ranges>)

Constructor.

=over 4

=item I<var>

Variable referenced, as L<Bi::Model::Var> derived object.

=item I<offsets>

Array ref of dimension offsets into I<var>, as L<Bi::Expression::Offset>
objects.

=item I<ranges>

Array ref of dimension ranges into I<var>, as L<Bi::Expression::Range>
objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $var = shift;
    my $offsets = shift;
    my $ranges = shift;
    
    # pre-conditions
    assert (defined $var && $var->isa('Bi::Model::Var')) if DEBUG;
    
    assert(!defined($offsets) || ref($offsets) eq 'ARRAY');
    assert(!defined($offsets) || scalar(@$offsets) == 0 || scalar(@$offsets) == $var->num_dims) if DEBUG;
    map { assert($_->isa('Bi::Expression::Offset')) if DEBUG } @$offsets;

    assert(!defined($ranges) || ref($ranges) eq 'ARRAY');
    assert(!defined($ranges) || scalar(@$ranges) == 0 || scalar(@$ranges) == $var->num_dims) if DEBUG;
    map { assert($_->isa('Bi::Expression::Range')) if DEBUG } @$ranges;

    if (!defined $offsets) {
        $offsets = [];
    }
    if (!defined $ranges) {
        $ranges = [];
    }
    
    my $self = {
        _var => $var,
        _offsets => $offsets,
        _ranges => $ranges
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
        _var => $self->get_var,
        _offsets => [ map { $_->clone } @{$self->get_offsets} ],
        _ranges => [ map { $_->clone } @{$self->get_ranges} ]
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_var>

Get the variable referenced, as L<Bi::Model::Var> derived object.

=cut
sub get_var {
    my $self = shift;
    return $self->{_var};
}

=item B<set_var>(I<var>)

Set the variable referenced.

=cut
sub set_var {
    my $self = shift;
    my $var = shift;
    $self->{_var} = $var;
}

=item B<get_offsets>

Get the array ref of dimension offsets into I<var>, as
L<Bi::Expression::Offset> objects.

=cut
sub get_offsets {
    my $self = shift;
    return $self->{_offsets};
}

=item B<set_offsets>(I<offsets>)

Set the array ref of dimension offsets into I<var>, as
L<Bi::Expression::Offset> objects.

=cut
sub set_offsets {
    my $self = shift;
    my $offsets = shift;
    
    # pre-conditions
    assert(!defined($offsets) || ref($offsets) eq 'ARRAY');
    assert(!defined($offsets) || scalar(@$offsets) == 0 || scalar(@$offsets) == $self->get_var->num_dims) if DEBUG;
    map { assert($_->isa('Bi::Expression::Offset')) if DEBUG } @$offsets;   
    
    $self->{_offsets} = $offsets;
}

=item B<get_ranges>

Get the array ref of dimension ranges into I<var>, as
L<Bi::Expression::Range> objects.

=cut
sub get_ranges {
    my $self = shift;
    return $self->{_ranges};
}

=item B<set_ranges>

Set the array ref of dimension ranges into I<var>, as
L<Bi::Expression::Range> objects.

=cut
sub set_ranges {
    my $self = shift;
    my $ranges = shift;
    
    # pre-conditions
    assert(!defined($ranges) || ref($ranges) eq 'ARRAY');
    assert(!defined($ranges) || scalar(@$ranges) == 0 || scalar(@$ranges) == $self->get_var->num_dims) if DEBUG;
    map { assert($_->isa('Bi::Expression::Range')) if DEBUG } @$ranges;
    
    $self->{_ranges} = $ranges;
}

=item B<num_dims>

Get the dimensionality of the expression.

=cut
sub num_dims {
    my $self = shift;

    return $self->get_var->num_dims - $self->num_offsets;
}

=item B<num_offsets>

The number of offsets.

=cut
sub num_offsets {
    my $self = shift;
    return scalar(@{$self->get_offsets});
}

=item B<no_offsets>

Are all offsets zero?

=cut
sub no_offset {
    my $self = shift;
    
    my $offset;
    my $result = 1;
    foreach $offset (@{$self->get_offsets}) {
        if ($offset->get_offset != 0) {
            $result = 0;
            last;
        }
    }
    
    return $result;
}

=item B<num_ranges>

The number of ranges.

=cut
sub num_ranges {
    my $self = shift;
    return scalar(@{$self->get_ranges});
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    @{$self->{_offsets}} = map { $_->accept($visitor, @args) } @{$self->get_offsets};
    @{$self->{_ranges}} = map { $_->accept($visitor, @args) } @{$self->get_ranges};

    return $visitor->visit($self, @args);
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_var == $obj->get_var &&
        Bi::Utility::equals($self->get_offsets, $obj->get_offsets) &&
        Bi::Utility::equals($self->get_ranges, $obj->get_ranges)); 
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
