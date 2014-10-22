=head1 NAME

Bi::Expression::InlineIdentifier - reference to inline expression.

=head1 SYNOPSIS

    use Bi::Expression::InlineIdentifier;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::InlineIdentifier;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

use Bi::Expression;

=item B<new>(I<inline>)

Constructor.

=over 4

=item I<inline> Inline expression, as L<Bi::Model::Inline> object.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $inline = shift;

    my $self = {
        _inline => $inline,
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

=item B<get_inline>

Get the inline expression being referenced, as L<Bi::Model::Inline> object.

=cut
sub get_inline {
    my $self = shift;
    return $self->{_inline};
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
        $self->get_inline->get_name eq $obj->get_inline->get_name); 
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
