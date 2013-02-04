=head1 NAME

Bi::Expression::Range - dimension range in a variable reference.

=head1 SYNOPSIS

    use Bi::Expression::Range;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Range;

use base 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<start>, I<end>)

Constructor.

=over 4

=item I<start>

L<Bi::Expression> giving starting index.

=item I<end>

L<Bi::Expression> giving ending index.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $start = shift;
    my $end = shift;
    
    my $self = {
        _start => $start,
        _end => $end,
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

=item B<get_start>

Get the starting index.

=cut
sub get_start {
    my $self = shift;
    return $self->{_start};
}

=item B<get_end>

Get the ending index.

=cut
sub get_end {
    my $self = shift;
    return $self->{_end};
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;
    
    $self->{_start} = $self->get_start->accept($visitor, @args);
    $self->{_end} = $self->get_end->accept($visitor, @args);
    
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
        $self->get_start->equals($obj->get_start) &&
        $self->get_end->equals($obj->get_end));
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
