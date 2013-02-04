=head1 NAME

Bi::Model::DimAlias - dimension alias.

=head1 SYNOPSIS

    use Bi::Model::DimAlias;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Model::DimAlias;

use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<name>, I<start>, I<end>)

Constructor.

=over 4

=item I<name>

Alias of the dimension.

=item I<start>

Starting index, if any.

=item I<end>

Ending index, if any.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $start = shift;
    my $end = shift;
    
    my $self = {
        _name => $name,
        _start => $start,
        _end => $end
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

=item B<get_name>

Get the name of the alias.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_start>

Get the starting index of the alias.

=cut
sub get_start {
    my $self = shift;
    return $self->{_start};
}

=item B<has_start>

Is there a starting index?

=cut
sub has_start {
	my $self = shift;
	return defined $self->{_start};
}

=item B<get_end>

Get the ending index of the alias.

=cut
sub get_end {
    my $self = shift;
    return $self->{_end};
}

=item B<has_end>

Is there an ending index?

=cut
sub has_end {
	my $self = shift;
	return defined $self->{_end};
}

=item B<num_dims>

Number of dimensions (always zero).

=cut
sub num_dims {
    my $self = shift;
    return 0;
}

=item B<gen_index>

Generate an index for this alias, as L<Bi::Expression::Index> object.

=cut
sub gen_index {
    my $self = shift;
    
    return new Bi::Expression::Index(new Bi::Expression::DimAliasIdentifier($self));
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
    
    return ref($obj) eq ref($self) && $self->get_name eq $obj->get_name;
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 2921 $ $Date: 2012-08-12 13:49:45 +0800 (Sun, 12 Aug 2012) $
