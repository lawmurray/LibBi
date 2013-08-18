=head1 NAME

Bi::Expression::VarIdentifier - reference to variable, with or without
dimension indexes.

=head1 SYNOPSIS

    use Bi::Expression::VarIdentifier;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::VarIdentifier;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<var>, I<indexes>)

Constructor.

=over 4

=item I<var>

Variable referenced, as L<Bi::Model::Var> derived object.

=item I<indexes>

Array ref of L<Bi::Expression::Index> or L<Bi::Expression::Range> objects
giving dimension indexes.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $var = shift;
    my $indexes = shift;
    
    # pre-conditions
    assert (defined $var && $var->isa('Bi::Model::Var')) if DEBUG;
    
    assert(!defined($indexes) || ref($indexes) eq 'ARRAY');
    assert(!defined($indexes) || scalar(@$indexes) == 0 || scalar(@$indexes) == scalar(@{$var->get_dims})) if DEBUG;
    map { assert($_->isa('Bi::Expression::Index') || $_->isa('Bi::Expression::Range')) if DEBUG } @$indexes;

    if (!defined $indexes) {
        $indexes = [];
    }
    
    my $self = {
        _var => $var,
        _indexes => $indexes
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
        _indexes => [ map { $_->clone } @{$self->get_indexes} ]
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

=item B<get_shape>

Get the shape of the result of the expression, as a L<Bi::Expression::Shape>
object.

=cut
sub get_shape {
    my $self = shift;

    if (@{$self->get_indexes}) {
        my $shape = [];
        for (my $i = 0; $i < @{$self->get_indexes}; ++$i) {
            if ($self->get_indexes->[$i]->is_range) {
                push(@$shape, $self->get_indexes->[$i]->get_size);
            }
        }
        return new Bi::Expression::Shape($shape);
    } else {
        return $self->get_var->get_shape;
    }
}

=item B<get_indexes>

Get the array ref of dimension indexes into I<var>, as
L<Bi::Expression::Index> and L<Bi::Expression::Range> objects.

=cut
sub get_indexes {
    my $self = shift;
    return $self->{_indexes};
}

=item B<set_indexes>(I<indexes>)

Set the array ref of dimension indexes into I<var>, as
L<Bi::Expression::Index> and L<Bi::Expression::Range> objects.

=cut
sub set_indexes {
    my $self = shift;
    my $indexes = shift;
    
    # pre-conditions
    assert(!defined($indexes) || ref($indexes) eq 'ARRAY');
    assert(!defined($indexes) || scalar(@$indexes) == 0 || scalar(@$indexes) == scalar(@{$self->get_var->get_dims})) if DEBUG;
    
    map { assert($_->isa('Bi::Expression::Index') || $_->isa('Bi::Expression::Range')) } @$indexes if DEBUG;   
    
    $self->{_indexes} = $indexes;
}

=item B<trivial_index>

Are the indexes trivial? This means that they are simply
L<Bi::Expression::DimAliasIdentifier> objects wrapped around the dimensions
of the variable, in the correct order.

=cut
sub trivial_index {
    my $self = shift;
    
    my $refs1 = $self->get_var->gen_indexes;
    my $refs2 = $self->get_indexes;
    
    return Bi::Utility::equals($refs1, $refs2);
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    $self = $visitor->visit_before($self, @args);
    @{$self->{_indexes}} = map { $_->accept($visitor, @args) } @{$self->get_indexes};

    return $visitor->visit_after($self, @args);
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_var->equals($obj->get_var) &&
        Bi::Utility::equals($self->get_indexes, $obj->get_indexes)); 
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
