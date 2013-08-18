=head1 NAME

Bi::Expression::Index - dimension index in a variable reference.

=head1 SYNOPSIS

    use Bi::Expression::Index;

=head1 INHERITS

L<Bi::Expression>

=head1 METHODS

=over 4

=cut

package Bi::Expression::Index;

use parent 'Bi::Expression';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<expr>)

Constructor.

=over 4

=item I<expr>

Index expression as L<Bi::Expression> object.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $expr = shift;
    
    assert(defined $expr) if DEBUG;
    
    my $self = {
        _expr => $expr
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

=item B<get_expr>

Get the index expression.

=cut
sub get_expr {
    my $self = shift;
    return $self->{_expr};
}

=item B<is_index>

Is this an index?

=cut
sub is_index {
    return 1;
}

=item B<is_range>

Is this a range?

=cut
sub is_range {
    return 0;
}

=item B<get_size>

Size of the range.

=cut
sub get_size {
    return 1;
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;
    
    $self = $visitor->visit_before($self, @args);
    $self->{_expr} = $self->get_expr->accept($visitor, @args);
    
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
        $self->get_expr->equals($obj->get_expr));
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
