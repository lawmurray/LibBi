=head1 NAME

Bi::Expression::DimAliasIdentifier - reference to dimension alias.

=head1 SYNOPSIS

    use Bi::Expression::DimAliasIdentifier;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::DimAliasIdentifier;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

=item B<new>(I<alias>)

Constructor.

=over 4

=item I<alias>

The dimension alias referenced, as a L<Bi::Model::DimAlias> object.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $alias = shift;
    
    assert ($alias->isa('Bi::Model::DimAlias')) if DEBUG;
    
    my $self = {
        _alias => $alias
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
    	_alias => $self->get_alias
    };
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<get_alias>

Get the dimension alias, as a L<Bi::Model::DimAlias> object.

=cut
sub get_alias {
    my $self = shift;
    return $self->{_alias};
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;
    
    my $new = $visitor->visit_before($self, @args);
    if (refaddr($new) == refaddr($self)) {
        $new = $visitor->visit_after($new, @args);
    }
    return $new;
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (
        ref($obj) eq ref($self) &&
        $self->get_alias->equals($obj->get_alias));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

