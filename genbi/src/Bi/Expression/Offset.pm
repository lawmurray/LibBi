=head1 NAME

Bi::Expression::Offset - dimension offset in a constant or variable
reference.

=head1 SYNOPSIS

    use Bi::Expression::Offset;

=head1 METHODS

=over 4

=cut

package Bi::Expression::Offset;

use base 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<alias>, I<offset>)

Constructor.

=over 4

=item I<alias>

Alias of the dimension.

=item I<offset>

Offset along the dimension, as an integer.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $alias = shift;
    my $offset = shift;
    
    my $self = {
        _alias => $alias,
        _offset => $offset,
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

=item B<get_alias>

Get the alias, as a string.

=cut
sub get_alias {
    my $self = shift;
    return $self->{_alias};
}

=item B<get_offset>

Get the offset.

=cut
sub get_offset {
    my $self = shift;
    return $self->{_offset};
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;
    
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
        $self->get_alias eq $obj->get_alias &&
        $self->get_offset == $obj->get_offset);
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
