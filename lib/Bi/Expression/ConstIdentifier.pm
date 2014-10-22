=head1 NAME

Bi::Expression::ConstIdentifier - reference to constant.

=head1 SYNOPSIS

    use Bi::Expression::ConstIdentifier

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::ConstIdentifier;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

use Bi::Expression;

=item B<new>(I<const>)

Constructor.

=over 4

=item I<const>

The constant referenced, as L<Bi::Model::Const> object.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $const = shift;

    my $self = {
        _const => $const,
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

=item B<get_const>

Get the constant referenced, as L<Bi::Model::Const> object.

=cut
sub get_const {
    my $self = shift;
    return $self->{_const};
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
	    $new = $visitor->visit_after($self, @args);
    }
    return $new;
}

=item B<equals>(I<obj>)

Does object equal I<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return (ref($obj) eq ref($self) &&
        $self->get_const->equals($obj->get_const)); 
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
